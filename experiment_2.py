# Wpływ parametru gamma na szerokość wiązki

import acoular as ac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from pathlib import Path
import time
import os
from Beamformer import Beamformer

# ==========================================
# KONFIGURACJA EKSPERYMENTU 2
# ==========================================

# Ścieżki do plików wejściowych (zgodne z poprzednimi krokami)
GEOM_DIR = Path("geom")
SIGNAL_DIR = Path("signal/dynamic")

# Wybrane parametry
geometries = ["rect_64", "sunflower_64"]
gammas = [-1, 1, 4, 50, 300]
trajectory_type = "circle"  # Najbardziej reprezentatywna

print(f"--- START EXPERIMENT 2: BEAMWIDTH ANALYSIS ---")
print(f"Geometries: {geometries}")
print(f"Gammas: {gammas}")
print("-" * 40)

for geom_name in geometries:
    for g in gammas:
        print(f"\n>>> PROCESSING: {geom_name} | Gamma = {g}")

        # Ścieżki do konkretnych plików
        geom_path = GEOM_DIR / f"{geom_name}.xml"
        wav_path = SIGNAL_DIR / f"{geom_name}_{trajectory_type}.wav"

        # Sprawdzenie czy pliki istnieją
        if not geom_path.exists():
            print(f"!!! ERROR: Missing geometry file: {geom_path}")
            continue
        if not wav_path.exists():
            print(f"!!! ERROR: Missing signal file: {wav_path}")
            continue

        try:
            bf = Beamformer(
                geom_path=geom_path,
                inputfile_path=wav_path,
                gamma=g,
                algorithm='BeamformerFunctional'
            )

            print(f"Done: {geom_name} g={g}")

        except Exception as e:
            print(f"!!! CRITICAL ERROR processing {geom_name} g={g}: {e}")

print("\n" + "=" * 40)
print("EXPERIMENT 2 COMPLETED.")