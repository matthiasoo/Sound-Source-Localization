from pathlib import Path
from SignalGenerator import DroneSignalGenerator, SignalRecorder

t_signal = 10
f_sample = 48000

drone_signal = DroneSignalGenerator(
    rpm_list=[15010, 14962, 13536, 13007],
    num_blades_per_rotor=2,
    sample_freq=f_sample,
    num_samples=int(f_sample * t_signal),
    rms=1
)

geom_folder = Path('geom')

files = [
    "2_mics.xml",
    "rect_4.xml",
    "rect_16.xml",
    "rect_64.xml",
    "ring_32.xml",
    "spiral_64.xml",
    "sunflower_64.xml",
]

print(f"--- STARTING SIGNAL GENERATION ---")
print(f"Parameters: {t_signal}s | {f_sample}Hz")
print(f"Geometries count: {len(files)}")
print(f"Trajectories per geometry: 3")
print("-" * 40)

for i, filename in enumerate(files):
    fpath = geom_folder / filename

    if not fpath.exists():
        print(f"!!! ERROR: Geometry file not found: {filename}. Skipping.")
        continue

    print(f"\n[{i + 1}/{len(files)}] Processing geometry: {filename}")

    try:
        recorder = SignalRecorder(fpath, drone_signal)

        recorder.run_linear()
        recorder.run_diagonal()
        recorder.run_circle()

    except Exception as e:
        print(f"!!! ERROR with {filename}: {e}")

print("\n" + "-"*40)
print("--- DONE ---")