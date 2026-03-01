import numpy as np




def build_simple_mosaic(frames_bgr, overlap_px: int = 80):
   if not frames_bgr:
       return None


   base = frames_bgr[0].copy()


   for nxt in frames_bgr[1:]:
       h = min(base.shape[0], nxt.shape[0])
       base = base[:h]
       nxt = nxt[:h]


       w1 = base.shape[1]
       w2 = nxt.shape[1]


       overlap = min(overlap_px, w1, w2)


       out = np.zeros((h, w1 + w2 - overlap, 3), dtype=np.uint8)
       out[:, :w1] = base


       for i in range(overlap):
           a = i / (overlap - 1) if overlap > 1 else 1.0
           left_col = base[:, w1 - overlap + i].astype(np.float32)
           right_col = nxt[:, i].astype(np.float32)
           out[:, w1 - overlap + i] = np.clip(
               (1 - a) * left_col + a * right_col,
               0, 255
           ).astype(np.uint8)


       out[:, w1:] = nxt[:, overlap:]
       base = out


   return base

