import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tinyinfer.api.server import create_app
from tinyinfer.runtime import InferenceRuntime, QueueFullError, WorkerUnavailableError


class FakeEngine:
    def __init__(self, delay=0.03, fail=False):
        self.requests = {}
        self.next_id = 0
        self.delay = delay
        self.fail = fail
        self.cleared = False
        self.threads = set()
        self.scheduler = SimpleNamespace(stats=lambda: {'waiting': 0, 'running': len(self.requests)})

    def add_request(self, prompt, params):
        self.threads.add(threading.get_ident())
        if prompt == 'invalid':
            raise ValueError('prompt is invalid')
        rid = self.next_id
        self.next_id += 1
        self.requests[rid] = SimpleNamespace(request_id=rid, prompt=prompt,
                                             output_text='done', num_generated=1)
        return rid

    def step(self):
        self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError('simulated model failure')

    def pop_finished(self, rid):
        return self.requests.pop(rid, None)

    def discard_request(self, rid):
        self.requests.pop(rid, None)

    def clear(self):
        self.requests.clear()
        self.cleared = True


def test_worker_failure_releases_all_waiters_and_rejects_new_requests():
    async def scenario():
        engine = FakeEngine(fail=True)
        runtime = InferenceRuntime(lambda: engine)
        await runtime.start()
        try:
            results = await asyncio.gather(*(runtime.generate('hello', None, 1) for _ in range(3)),
                                           return_exceptions=True)
            assert all(isinstance(r, WorkerUnavailableError) for r in results)
            assert not runtime.healthy
            with pytest.raises(WorkerUnavailableError):
                await runtime.generate('later', None, 1)
        finally:
            await runtime.close()
        assert engine.cleared and not engine.requests and not runtime.jobs
    asyncio.run(scenario())


def test_timeout_backpressure_cleanup_and_recovery():
    async def scenario():
        engine = FakeEngine(delay=0.1)
        runtime = InferenceRuntime(lambda: engine, max_pending=1)
        await runtime.start()
        try:
            first = asyncio.create_task(runtime.generate('first', None, 0.01))
            await asyncio.sleep(0)
            with pytest.raises(QueueFullError):
                await runtime.generate('second', None, 1)
            with pytest.raises(TimeoutError):
                await first
            # Timed-out work stays counted until the current model call returns.
            deadline = asyncio.get_running_loop().time() + 2
            while runtime.jobs and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert not runtime.jobs and not engine.requests
            assert (await runtime.generate('third', None, 1)).output_text == 'done'
            assert len(engine.threads) == 1
            assert threading.get_ident() not in engine.threads
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_cancellation_before_execution_does_not_enter_engine():
    async def scenario():
        engine = FakeEngine()
        runtime = InferenceRuntime(lambda: engine)
        await runtime.start()
        try:
            task = asyncio.create_task(runtime.generate('cancel', None, 1))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.01)
            assert not engine.requests
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_api_validation_success_and_failure_health():
    with TestClient(create_app(lambda: FakeEngine())) as client:
        assert client.get('/health').status_code == 200
        for body in [{'prompt': ''}, {'prompt': 'x', 'max_tokens': 0},
                     {'prompt': 'x', 'top_p': 0}, {'prompt': 'x', 'temperature': -1}]:
            assert client.post('/generate', json=body).status_code == 422
        assert client.post('/generate', json={'prompt': 'invalid'}).status_code == 422
        response = client.post('/generate', json={'prompt': 'hello'})
        assert response.status_code == 200
        assert response.json()['text'] == 'done'
    with TestClient(create_app(lambda: FakeEngine(fail=True))) as client:
        assert client.post('/generate', json={'prompt': 'hello'}).status_code == 503
        assert client.get('/health').status_code == 503
    with TestClient(create_app(lambda: FakeEngine(delay=0.1), request_timeout=0.01)) as client:
        assert client.post('/generate', json={'prompt': 'hello'}).status_code == 504


def test_health_remains_responsive_while_inference_is_blocked():
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    engine = FakeEngine()
    def blocked_step():
        entered.set()
        assert release.wait(3)
    engine.step = blocked_step
    with TestClient(create_app(lambda: engine)) as client, ThreadPoolExecutor() as pool:
        response = pool.submit(client.post, '/generate', json={'prompt': 'hello'})
        assert entered.wait(2)
        try:
            # Inference is held by an event, so success here proves event-loop liveness.
            health = pool.submit(client.get, '/health').result(timeout=1)
            assert health.status_code == 200
        finally:
            release.set()
        assert response.result(timeout=2).status_code == 200


def test_inflight_cancellation_releases_capacity_after_step():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        engine = FakeEngine()
        def blocked_step():
            entered.set()
            assert release.wait(2)
        engine.step = blocked_step
        runtime = InferenceRuntime(lambda: engine, max_pending=1)
        await runtime.start()
        task = asyncio.create_task(runtime.generate('cancel-active', None, 2))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(QueueFullError):
                await runtime.generate('too-early', None, 1)
            release.set()
            deadline = asyncio.get_running_loop().time() + 1
            while runtime.jobs and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert not runtime.jobs and not engine.requests
            assert (await runtime.generate('after-cleanup', None, 1)).output_text == 'done'
        finally:
            release.set()
            await runtime.close()
    asyncio.run(scenario())
