from .config import MODEL_PATH, PROJECT_ROOT
from .status import RequestStatus

__all__ = [
    "MODEL_PATH",
    "PROJECT_ROOT",
    "Request",
    "RequestStatus",
    "load_model",
    "load_tokenizer",
    "sample_next_token",
]


def __getattr__(name):
    if name == "Request":
        from .request import Request

        return Request
    if name in {"load_model", "load_tokenizer"}:
        from .model_loader import load_model, load_tokenizer

        return {"load_model": load_model, "load_tokenizer": load_tokenizer}[name]
    if name == "sample_next_token":
        from .sampler import sample_next_token

        return sample_next_token
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
