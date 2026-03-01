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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    cfg = load_config(parser.parse_args().config)

    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = OutputWriter(out_dir, cfg)
    timer = Timer()

    images = load_image_strip(cfg["input"]["image_dir"])
    imu = load_imu_csv(cfg["input"]["imu_csv"])

    processed_frames = []
    frame_data = []

    for i, img in enumerate(images):

        if timer.now() > cfg["runtime"]["max_processing_s"]:
            break

        try:
            img_r, gray, enh, den, edges, canny = preprocess_frame(img, cfg)

            sh_mask = shadow_mask_from_gray(enh, cfg)
            hazards = detect_hazards(enh, edges, cfg)

            heights = estimate_heights_from_shadows(
                enh,
                sh_mask,
                hazards,
                cfg["mission"]["solar_elevation_deg"],
                cfg["mission"]["solar_azimuth_deg"],
                cfg["scene"]["scene_width_m"],
                cfg
            )

            tr = cfg["traversability"]
            rows = int(tr["grid_h_m"] / tr["grid_cell_m"])
            cols = int(tr["grid_w_m"] / tr["grid_cell_m"])

            sh_grid = shadow_coverage_grid(sh_mask, rows, cols)

            processed_frames.append(img_r)

            frame_data.append({
                "hazards": hazards,
                "heights": heights,
                "shadow_grid": sh_grid.tolist()
            })

            if cfg["output"].get("save_debug"):
                writer.save_debug_frame(i, img_r, sh_mask, hazards)

        except Exception as e:
            print(f"Skipping frame {i} due to error: {e}")

    mosaic = build_simple_mosaic(processed_frames)
    final_grid = build_traversability_grid(processed_frames, frame_data, cfg)

    writer.save_final(mosaic, final_grid, frame_data, imu, timer.summary())

    if cfg["output"].get("make_zip"):
        writer.make_downlink_zip()


if __name__ == "__main__":
    main()
