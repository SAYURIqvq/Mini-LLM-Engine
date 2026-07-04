import pytest

torch = pytest.importorskip("torch")

from tinyinfer.core.sampler import sample_next_token


def test_sample_next_token_greedy_when_temperature_zero():
    logits = torch.tensor([[0.1, 3.0, 0.2]])

    token = sample_next_token(logits, temperature=0)

    assert token.item() == 1


def test_sample_next_token_respects_top_p_candidate_set(monkeypatch):
    logits = torch.tensor([[10.0, 9.0, 1.0]])
    captured = {}

    def fake_multinomial(probs, num_samples):
        captured["probs"] = probs
        return torch.tensor([[0]])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    sample_next_token(logits, temperature=1.0, top_p=0.8)

    assert captured["probs"][0, 2].item() == 0
    assert captured["probs"][0, 0].item() > 0
    assert captured["probs"][0, 1].item() > 0
