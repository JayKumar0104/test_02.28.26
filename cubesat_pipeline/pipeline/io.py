import csv
from pathlib import Path
import cv2




def load_image_strip(image_dir: str):
   p = Path(image_dir)
   if not p.exists():
       raise FileNotFoundError(f"image_dir not found: {image_dir}")


   exts = {".jpg", ".jpeg", ".png", ".bmp"}
   files = sorted([x for x in p.iterdir() if x.suffix.lower() in exts])


   if not files:
       raise RuntimeError(f"No images found in {image_dir}")


   images = []
   for f in files:
       img = cv2.imread(str(f), cv2.IMREAD_COLOR)
       if img is None:
           raise RuntimeError(f"Failed to load image: {f}")
       images.append(img)


   return images




def load_imu_csv(imu_csv: str):
   p = Path(imu_csv)


   if not p.exists():
       return {
           "rows": [],
           "columns": [],
           "note": "imu.csv not found (ok for demo)"
       }


   with p.open("r", encoding="utf-8", newline="") as f:
       reader = csv.DictReader(f)
       rows = list(reader)
       cols = reader.fieldnames or []


   return {
       "rows": rows,
       "columns": cols,
       "note": "loaded"
   }

