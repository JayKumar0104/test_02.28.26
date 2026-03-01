import argparse
from pathlib import Path

from pipeline.config import load_config
from pipeline.io import load_image_strip, load_imu_csv
from pipeline.timing import Timer
from pipeline.preprocess import preprocess_frame
from pipeline.shadow import shadow_mask_from_gray, shadow_coverage_grid
from pipeline.hazards import detect_hazards
from pipeline.geometry import estimate_heights_from_shadows
from pipeline.mosaic import build_simple_mosaic
from pipeline.traversability import build_traversability_grid
from pipeline.outputs import OutputWriter


def run(cfg: dict):
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = OutputWriter(out_dir=out_dir, cfg=cfg)
    timer = Timer()

    images = load_image_strip(cfg["input"]["image_dir"])
    imu = load_imu_csv(cfg["input"]["imu_csv"])

    tr = cfg["traversability"]
    rows = max(1, int(tr["grid_h_m"] / tr["grid_cell_m"]))
    cols = max(1, int(tr["grid_w_m"] / tr["grid_cell_m"]))

    mode = "normal"
    max_s = cfg["runtime"]["max_processing_s"]

    if cfg["runtime"]["degrade"]["enable"] and images:
        timer.mark("probe")
        img_r, gray, enh, den, edges, _ = preprocess_frame(images[0], cfg)
        sh = shadow_mask_from_gray(enh, cfg)
        hz = detect_hazards(enh, edges, cfg)
        estimate_heights_from_shadows(
            enhanced_gray=enh,
            shadow_mask=sh,
            hazards=hz,
            solar_elev_deg=cfg["mission"]["solar_elevation_deg"],
            solar_azimuth_deg=cfg["mission"]["solar_azimuth_deg"],
            scene_width_m=cfg["scene"]["scene_width_m"],
            cfg=cfg,
        )
        probe_time = timer.elapsed_since("probe")
        if probe_time * len(images) * 2.8 > max_s:
            mode = "degraded"

    writer.write_run_metadata(mode)

    frames = []
    per_frame = []

    for i, img in enumerate(images):
        if timer.now() > max_s:
            break

        img_r, gray, enh, den, edges, _ = preprocess_frame(img, cfg)

        sh_mask = shadow_mask_from_gray(enh, cfg)
        hazards = detect_hazards(enh, edges, cfg)

        heights = estimate_heights_from_shadows(
            enhanced_gray=enh,
            shadow_mask=sh_mask,
            hazards=hazards,
            solar_elev_deg=cfg["mission"]["solar_elevation_deg"],
            solar_azimuth_deg=cfg["mission"]["solar_azimuth_deg"],
            scene_width_m=cfg["scene"]["scene_width_m"],
            cfg=cfg,
        )

        sh_grid = shadow_coverage_grid(sh_mask, rows, cols)

        frames.append(img_r)
        per_frame.append({
            "index": i,
            "hazards": hazards,
            "heights": heights,
            "shadow_grid": sh_grid.tolist()
        })

    mosaic = build_simple_mosaic(frames) if frames else None
    grid = build_traversability_grid(frames, per_frame, cfg)

    writer.save_final(
        mosaic_bgr=mosaic,
        grid=grid,
        per_frame=per_frame,
        imu=imu,
        timing_summary=timer.summary(),
    )

    if cfg["output"]["make_zip"]:
        writer.make_downlink_zip()

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = run(cfg)
    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

