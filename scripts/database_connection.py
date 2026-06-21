from pathlib import Path
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "database"
DB_PATH = DB_DIR / "github_herd.db"

def get_engine():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")
