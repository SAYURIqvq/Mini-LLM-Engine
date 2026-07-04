__all__ = ["TinyInferEngine"]


def __getattr__(name):
    if name == "TinyInferEngine":
        from .engine import TinyInferEngine

        return TinyInferEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
