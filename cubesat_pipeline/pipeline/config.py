from pathlib import Path
import yaml




def load_config(path: str) -> dict:
   p = Path(path)
   if not p.exists():
       raise FileNotFoundError(f"Config not found: {path}")


   with p.open("r", encoding="utf-8") as f:
       return yaml.safe_load(f) or {}

