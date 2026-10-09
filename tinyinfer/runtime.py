"""One inference thread; asyncio owns admission, deadlines and client futures."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


class QueueFullError(Exception):
    pass


class WorkerUnavailableError(Exception):
    pass


@dataclass
class Job:
    prompt: str
    params: Any
    future: asyncio.Future
    request_id: int | None = None


class InferenceRuntime:
    def __init__(self, engine_factory, max_pending=64):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.engine_factory = engine_factory
        self.max_pending = max_pending
        self.jobs = []
        self.wakeup = asyncio.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.engine = None
        self.task = None
        self.closing = False
        self.failure = None
        self.stats = {"waiting": 0, "running": 0}

    async def _call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def start(self):
        try:
            self.engine = await self._call(self.engine_factory)
        except BaseException:
            self.executor.shutdown(wait=False)
            raise
        self.task = asyncio.create_task(self._run())

    @property
    def healthy(self):
        return (not self.closing and self.failure is None and
                self.task is not None and not self.task.done())

    async def generate(self, prompt, params, timeout):
        if not self.healthy:
            raise WorkerUnavailableError("Inference worker is unavailable")
        if len(self.jobs) >= self.max_pending:
            raise QueueFullError("Inference queue is full")
        job = Job(prompt, params, asyncio.get_running_loop().create_future())
        self.jobs.append(job)
        self.wakeup.set()
        try:
            return await asyncio.wait_for(job.future, timeout=timeout)
        finally:
            # A timeout/disconnect cancels the future. The worker releases its KV
            # cache after the current non-preemptible model step returns.
            if not job.future.done():
                job.future.cancel()
            self.wakeup.set()

    async def _run(self):
        try:
            while not self.closing:
                self.wakeup.clear()
                for job in list(self.jobs):
                    if job.future.cancelled():
                        if job.request_id is not None:
                            await self._call(self.engine.discard_request, job.request_id)
                        self.jobs.remove(job)
                    elif job.request_id is None:
                        try:
                            job.request_id = await self._call(self.engine.add_request, job.prompt, job.params)
                        except ValueError as exc:
                            if not job.future.done():
                                job.future.set_exception(exc)
                            self.jobs.remove(job)
                if self.jobs:
                    await self._call(self.engine.step)
                    for job in list(self.jobs):
                        result = await self._call(self.engine.pop_finished, job.request_id)
                        if result is not None:
                            if not job.future.done():
                                job.future.set_result(result)
                            self.jobs.remove(job)
                    self.stats = await self._call(self.engine.scheduler.stats)
                else:
                    self.stats = {"waiting": 0, "running": 0}
                    await self.wakeup.wait()
        except Exception as exc:
            logging.getLogger(__name__).exception("Inference worker failed")
            self.failure = str(exc)
        finally:
            for job in self.jobs:
                if not job.future.done():
                    job.future.set_exception(WorkerUnavailableError("Inference worker stopped"))
            # Cleanup executes on the same thread as inference, after its last step.
            try:
                if self.engine is not None:
                    await self._call(self.engine.clear)
            finally:
                self.jobs.clear()
                self.stats = {"waiting": 0, "running": 0}

    async def close(self):
        self.closing = True
        self.wakeup.set()
        try:
            if self.task is not None:
                await self.task
        finally:
            self.executor.shutdown(wait=False)
