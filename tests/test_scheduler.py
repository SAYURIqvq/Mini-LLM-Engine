import pytest

from tinyinfer.core.status import RequestStatus
from tinyinfer.scheduler.continuous_batch import ContinuousBatchScheduler


class FakeRequest:
    def __init__(self, request_id: int):
        self.request_id = request_id
        self.status = RequestStatus.WAITING

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def finish(self, text: str):
        self.output_text = text
        self.status = RequestStatus.FINISHED


def make_request(request_id: int) -> FakeRequest:
    return FakeRequest(request_id)


def test_scheduler_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        ContinuousBatchScheduler(max_batch_size=0)


def test_scheduler_promotes_waiting_requests_up_to_capacity():
    scheduler = ContinuousBatchScheduler(max_batch_size=2)
    requests = [make_request(i) for i in range(3)]

    for request in requests:
        scheduler.add_request(request)

    batch = scheduler.schedule()

    assert [request.request_id for request in batch] == [0, 1]
    assert all(request.status == RequestStatus.RUNNING for request in batch)
    assert scheduler.stats() == {"waiting": 1, "running": 2, "max_batch_size": 2}


def test_scheduler_evicts_finished_requests_and_fills_slots():
    scheduler = ContinuousBatchScheduler(max_batch_size=2)
    first, second, third = [make_request(i) for i in range(3)]

    for request in [first, second, third]:
        scheduler.add_request(request)
    scheduler.schedule()

    first.finish("done")
    batch = scheduler.schedule()

    assert [request.request_id for request in batch] == [1, 2]
    assert scheduler.num_waiting == 0
    assert scheduler.num_running == 2
