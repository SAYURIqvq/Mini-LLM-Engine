import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## model path config
## defaults to PROJECT_ROOT/models/Qwen2.5-1.5B-Instruct 
MODEL_PATH = os.environ.get(
    "TINYINFER_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),
)
