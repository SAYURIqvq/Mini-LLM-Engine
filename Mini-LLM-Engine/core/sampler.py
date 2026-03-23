import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """
    Sample the next token from model output logits.

    logits: [batch_size, vocab_size] logits at the last position
    temperature: higher = more random, lower = more deterministic
    top_p: nucleus sampling, only sample from tokens whose cumulative probability <= top_p
    """
    # temperature scaling
    if temperature <= 0:
        # greedy: pick the most probable token
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # top_p (nucleus) sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # mask out tokens with cumulative probability exceeding top_p
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # restore original order
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # sample from the probability distribution
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
