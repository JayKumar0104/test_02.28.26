# MIT BWSI CubeSat Lunar Terrain Mapping Pipeline

Competition-ready terrain analysis pipeline for the MIT BWSI Build a CubeSat Challenge.

## Setup

pip install -r requirements.txt

## Run

python main.py --config config.yaml

Outputs appear in:
out/

Key outputs:
- preview_mosaic.jpg
- traversability_heatmap.png
- shadow_coverage.png
- results.json
- downlink_<runid>.zip

IMU file (data/imu.csv) is optional.
