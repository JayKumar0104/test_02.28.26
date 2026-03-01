import math
import numpy as np


def _shadow_length_px(shadow_mask_u8, x, y, r_px, azimuth_deg=None):
    h, w = shadow_mask_u8.shape[:2]
    mask = (shadow_mask_u8 > 0).astype(np.uint8)

    def march(dx, dy):
        run = 0
        max_t = int(r_px) + 250

        for t in range(int(r_px), max_t):
            xi = int(round(x + dx * t))
            yi = int(round(y + dy * t))

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                break

            if mask[yi, xi] == 1:
                run += 1
            else:
                if run > 0:
                    break

        return run

    if azimuth_deg is not None:
        ang = math.radians(float(azimuth_deg))
        dx = math.cos(ang)
        dy = math.sin(ang)
        return float(march(dx, dy))

    return 0.0


def estimate_heights_from_shadows(
    enhanced_gray,
    shadow_mask,
    hazards,
    solar_elev_deg,
    solar_azimuth_deg,
    scene_width_m,
    cfg
):
    img_w = enhanced_gray.shape[1]
    m_per_px = float(scene_width_m) / max(1.0, float(img_w))

    elev = math.radians(float(solar_elev_deg))
    tan_e = math.tan(elev)

    out = []

    for hz in hazards:
        x, y, r = hz["x"], hz["y"], hz["r"]

        L_px = _shadow_length_px(
            shadow_mask,
            x, y, r,
            azimuth_deg=solar_azimuth_deg
        )

        L_m = L_px * m_per_px
        h_m = L_m * tan_e   # ORIGINAL FORMULA (no tilt correction)

        out.append({
            "x": x,
            "y": y,
            "r": r,
            "shadow_len_px": round(float(L_px), 2),
            "height_m": round(float(h_m), 4),
        })

    return out
