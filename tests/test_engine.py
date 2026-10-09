from types import SimpleNamespace
import asyncio
import torch
import pytest
from tinyinfer import engine as module


class Model:
    config = SimpleNamespace(max_position_embeddings=16)
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor([[[0., 2., 0.]]]), past_key_values=('cache',))


class Tokenizer:
    eos_token_id = 2
    def encode(self, prompt, return_tensors):
        return torch.ones((1, len(prompt)), dtype=torch.long)
    def decode(self, ids, skip_special_tokens):
        return 'result'


def make_engine(monkeypatch):
    monkeypatch.setattr(module, 'load_model', lambda *args: Model())
    monkeypatch.setattr(module, 'load_tokenizer', lambda *args: Tokenizer())
    return module.TinyInferEngine('fake', max_input_tokens=8)


def test_engine_finishes_and_releases_cache(monkeypatch):
    engine = make_engine(monkeypatch)
    rid = engine.add_request('hi', module.SamplingParams(temperature=0, max_tokens=2))
    engine.step()
    assert engine._requests[rid].past_key_values is not None
    engine.step()
    result = engine.pop_finished(rid)
    assert result.num_generated == 2 and result.past_key_values is None
    assert not engine._requests and not engine._finished and not engine.scheduler.has_pending()


def test_invalid_prompt_and_context_budget(monkeypatch):
    engine = make_engine(monkeypatch)
    for prompt, params in [(' ', module.SamplingParams()),
                           ('123456789', module.SamplingParams()),
                           ('12345678', module.SamplingParams(max_tokens=10)),
                           ('hi', module.SamplingParams(max_tokens=0))]:
        with pytest.raises(ValueError):
            engine.add_request(prompt, params)
    assert not engine._requests


def test_discard_removes_queued_and_running_requests(monkeypatch):
    engine = make_engine(monkeypatch)
    first = engine.add_request('a', module.SamplingParams(max_tokens=5))
    engine.step()
    second = engine.add_request('b', module.SamplingParams(max_tokens=5))
    engine.discard_request(first)
    engine.discard_request(second)
    assert not engine.scheduler.has_pending() and not engine._requests
