import numpy as np


def build_traversability_grid(frames, per_frame, cfg: dict):

    tr = cfg["traversability"]
    grid_cell_m = float(tr["grid_cell_m"])
    grid_w_m = float(tr["grid_w_m"])
    grid_h_m = float(tr["grid_h_m"])

    overlap_px = 80

    if not frames:
        return {"rows": 0, "cols": 0, "cell_m": grid_cell_m, "risk": []}

    H, W = frames[0].shape[:2]

    shift_px = W - overlap_px
    total_width_px = W + (len(frames) - 1) * shift_px
    total_grid_w_m = grid_w_m * (total_width_px / W)

    rows = max(1, int(grid_h_m / grid_cell_m))
    cols = max(1, int(total_grid_w_m / grid_cell_m))

    hazard_risk = np.zeros((rows, cols), dtype=np.float32)
    shadow_risk = np.zeros((rows, cols), dtype=np.float32)
    slope_risk = np.zeros((rows, cols), dtype=np.float32)
    shadow_frame_count = np.zeros((rows, cols), dtype=np.float32)

    for frame_index, fr in enumerate(per_frame):

        x_offset_px = frame_index * shift_px

        col_start = int((x_offset_px / total_width_px) * cols)

        for hz in fr.get("hazards", []):
            x_global = hz["x"] + x_offset_px
            c = min(cols - 1, max(0, int((x_global / total_width_px) * cols)))
            r = min(rows - 1, max(0, int((hz["y"] / H) * rows)))
            hazard_risk[r, c] += (0.5 + 0.5 * hz.get("score", 0.0))

        sg = fr.get("shadow_grid")
        if sg is not None:
            sg_arr = np.array(sg, dtype=np.float32)
            f_cols = sg_arr.shape[1]
            col_end = min(cols, col_start + f_cols)
            usable = col_end - col_start

            if usable > 0:
                shadow_risk[:, col_start:col_end] += sg_arr[:, :usable]
                shadow_frame_count[:, col_start:col_end] += 1

        for ht in fr.get("heights", []):
            x_global = ht["x"] + x_offset_px
            c = min(cols - 1, max(0, int((x_global / total_width_px) * cols)))
            r = min(rows - 1, max(0, int((ht["y"] / H) * rows)))
            slope_risk[r, c] += abs(ht.get("height_m", 0.0))

    if hazard_risk.max() > 0:
        hazard_risk /= hazard_risk.max()

    if slope_risk.max() > 0:
        slope_risk /= slope_risk.max()

    shadow_risk = np.divide(
        shadow_risk,
        shadow_frame_count,
        out=np.zeros_like(shadow_risk),
        where=shadow_frame_count != 0
    )

    w_h = float(tr.get("w_hazard", 1.0))
    w_s = float(tr.get("w_shadow", 0.7))
    w_sl = float(tr.get("w_slope", 0.6))

    risk = (w_h * hazard_risk +
            w_s * shadow_risk +
            w_sl * slope_risk) / max(1e-6, (w_h + w_s + w_sl))

    return {
        "rows": rows,
        "cols": cols,
        "cell_m": grid_cell_m,
        "risk": np.clip(risk, 0, 1).tolist(),
        "components": {
            "hazard": hazard_risk.tolist(),
            "shadow": shadow_risk.tolist(),
            "slope": slope_risk.tolist()
        }
    }
