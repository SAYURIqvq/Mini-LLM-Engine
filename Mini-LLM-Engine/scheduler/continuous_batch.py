from collections import deque
from typing import Deque, List, Optional
from ..core.request import Request, RequestStatus


class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size

        self.waiting: Deque[Request] = deque()  # requests waiting to be scheduled
        self.running: List[Request] = []        # requests currently being processed

    def add_request(self, request: Request):
        """Add a new request to the waiting queue."""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def schedule(self) -> List[Request]:
        """
        Called each step to determine the current batch.

        1. Remove finished requests from the running batch
        2. Fill empty slots with waiting requests (up to max_batch_size)
        3. Return the current active batch
        """
        # remove finished requests
        self.running = [r for r in self.running if not r.is_finished]

        # fill empty slots from waiting queue
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running.append(request)

        return self.running

    def has_pending(self) -> bool:
        """Check if there are any unfinished requests."""
        return len(self.waiting) > 0 or len(self.running) > 0

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)
