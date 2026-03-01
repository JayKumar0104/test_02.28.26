import numpy as np
import cv2




def shadow_mask_from_gray(gray_or_enhanced, cfg: dict):
   percentile = float(cfg["shadow"].get("value_percentile", 12))
   thresh = int(np.percentile(gray_or_enhanced, percentile))


   mask = (gray_or_enhanced < thresh).astype(np.uint8) * 255


   k = int(cfg["shadow"].get("morph_kernel", 5))
   k = max(3, k)
   if k % 2 == 0:
       k += 1


   kernel = np.ones((k, k), np.uint8)


   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


   return mask




def shadow_coverage_grid(shadow_mask_u8, rows: int, cols: int):
   H, W = shadow_mask_u8.shape[:2]
   mask01 = (shadow_mask_u8 > 0).astype(np.float32)


   grid = np.zeros((rows, cols), dtype=np.float32)


   for r in range(rows):
       y0 = int(r * H / rows)
       y1 = int((r + 1) * H / rows)


       for c in range(cols):
           x0 = int(c * W / cols)
           x1 = int((c + 1) * W / cols)


           cell = mask01[y0:y1, x0:x1]
           grid[r, c] = float(cell.mean()) if cell.size else 0.0


   return grid

