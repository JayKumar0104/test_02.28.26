import cv2
import numpy as np




def _create_blob_detector(params):
   if hasattr(cv2, "SimpleBlobDetector_create"):
       return cv2.SimpleBlobDetector_create(params)


   if hasattr(cv2, "SimpleBlobDetector"):
       return cv2.SimpleBlobDetector(params)


   raise RuntimeError("OpenCV build missing SimpleBlobDetector APIs.")




def _blob_params(cfg: dict):
   params = cv2.SimpleBlobDetector_Params()


   params.minThreshold = 5
   params.maxThreshold = 220
   params.thresholdStep = 5


   params.filterByArea = True
   params.minArea = float(cfg["hazards"].get("min_area_px", 80))
   params.maxArea = float(cfg["hazards"].get("max_area_px", 8000))


   params.filterByCircularity = True
   params.minCircularity = float(cfg["hazards"].get("min_circularity", 0.25))


   params.filterByConvexity = False
   params.filterByInertia = False


   params.filterByColor = True
   params.blobColor = 0


   return params




def detect_hazards(enhanced_gray, edges, cfg: dict):
   params = _blob_params(cfg)
   detector = _create_blob_detector(params)


   keypoints = detector.detect(enhanced_gray)


   hazards = []
   h, w = enhanced_gray.shape[:2]
   yy, xx = np.ogrid[:h, :w]


   for kp in keypoints:
       x, y = int(kp.pt[0]), int(kp.pt[1])
       r = max(2, int(kp.size / 2))


       dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
       ring = (dist >= r * 0.8) & (dist <= r * 1.2)


       edge_strength = (
           float(edges[ring].mean()) / 255.0
           if ring.sum() > 20 else 0.0
       )


       hazards.append({
           "x": x,
           "y": y,
           "r": r,
           "score": round(edge_strength, 3)
       })


   return hazards

