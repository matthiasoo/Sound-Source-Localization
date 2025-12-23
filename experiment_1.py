# The influence of microphone array geometry on the quality of sound source localization

from pathlib import Path
from Beamformer import Beamformer
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

GAMMA_VALUE = 10

geometries = [
    "2_mics.xml",
    "rect_4.xml",
    "rect_16.xml",
    "rect_64.xml",
    "ring_32.xml",
    "spiral_64.xml",
    "sunflower_64.xml"
]

trajectories = [
    "linear",
    "diagonal",
    "circle"
]

geom_folder = Path('geom')
signal_folder = Path('signal/dynamic')

print(f"--- STARTING BEAMFORMER (Gamma={GAMMA_VALUE}) ---")
print(f"Geometries count: {len(geometries)}")
print(f"Trajectories count: {len(trajectories)}")
print("-" * 50)

total_tasks = len(geometries) * len(trajectories)
current_task = 0

for geom_file in geometries:
    geom_path = geom_folder / geom_file

    if not geom_path.exists():
        print(f"!!! ERROR: File not found: {geom_file}")
        continue

    geom_stem = geom_path.stem

    for traj in trajectories:
        current_task += 1

        wav_name = f"{geom_stem}_{traj}.wav"
        wav_path = signal_folder / wav_name

        if not wav_path.exists():
            print(f"[{current_task}/{total_tasks}] SKIPPING: No such file: {wav_name}")
            continue

        print(f"\n[{current_task}/{total_tasks}] PROCESSING: Geometry={geom_stem} | Trajectory={traj}")

        try:
            Beamformer(
                geom_path=str(geom_path),
                inputfile_path=str(wav_path),
                gamma=GAMMA_VALUE
            )
        except Exception as e:
            print(f"!!! ERROR with {wav_name}: {e}")

print("\n" + "-" * 50)
print("--- DONE ---")