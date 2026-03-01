import json, time, uuid, zipfile, cv2, numpy as np
from pathlib import Path


class OutputWriter:
    def __init__(self, out_dir: Path, cfg: dict):
        self.out_dir = out_dir
        self.cfg = cfg
        self.run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.debug_dir = out_dir / "debug_frames"
        self.debug_dir.mkdir(parents=True, exist_ok=True)


    def save_debug_frame(self, idx, img_bgr, shadow_mask, hazards):
        # Create visual overlay for hazards
        overlay = img_bgr.copy()
        for h in hazards:
            cv2.circle(overlay, (h['x'], h['y']), h['r'], (0, 255, 0), 2)
        
        fname = self.debug_dir / f"frame_{idx:03d}_overlay.jpg"
        cv2.imwrite(str(fname), overlay)
        cv2.imwrite(str(self.debug_dir / f"frame_{idx:03d}_shadow.png"), shadow_mask)


    def save_final(self, mosaic_bgr, grid, per_frame, imu, timing):
        if mosaic_bgr is not None:
            cv2.imwrite(str(self.out_dir / "preview_mosaic.jpg"), mosaic_bgr)


        # Save heatmap and shadow map
        for name, data in [("traversability_heatmap", grid["risk"]), ("shadow_coverage", grid["components"]["shadow"])]:
            arr = np.array(data, dtype=np.float32)
            res = cv2.resize((arr * 255).astype(np.uint8), (800, 400), interpolation=cv2.INTER_NEAREST)
            color = cv2.applyColorMap(res, cv2.COLORMAP_JET if "heat" in name else cv2.COLORMAP_BONE)
            cv2.imwrite(str(self.out_dir / f"{name}.png"), color)


        results = {"run_id": self.run_id, "config": self.cfg, "timing": timing, "grid": grid}
        (self.out_dir / "results.json").write_text(json.dumps(results, indent=2))


    def make_downlink_zip(self):
        zip_path = self.out_dir / f"downlink_{self.run_id}.zip"
        files = ["preview_mosaic.jpg", "traversability_heatmap.png", "shadow_coverage.png", "results.json"]
        with zipfile.ZipFile(zip_path, "w") as z:
            for f in files:
                if (self.out_dir / f).exists(): z.write(self.out_dir / f, arcname=f)

