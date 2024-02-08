from pathlib import Path

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
CURRENT_MODEL_PATH = MODEL_DIR / "current.pkl"
