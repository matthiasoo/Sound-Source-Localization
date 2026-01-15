from Beamformer import Beamformer3D
from pathlib import Path
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

geometries = [
    "cube_14.xml"
]

geom_folder = Path('geom_3d')
signal_folder = Path('signal/dynamic')

geom_path = geom_folder / "cube_14.xml"
wav_path = signal_folder / "cube_14_helix.wav"

Beamformer3D(
    geom_path=str(geom_path),
    inputfile_path=str(wav_path),
    gamma=10
)