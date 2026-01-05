import cv2
import os
from pathlib import Path

# --- KONFIGURACJA ---
RESULTS_DIR = Path("results")
# Zapisujemy w folderze dedykowanym dla Eksperymentu 2
OUTPUT_DIR = Path("analysis", "SNAPSHOTS_gamma")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TIME = 5.0
FPS = 30
FRAME_NO = int(TARGET_TIME * FPS)

# Parametry Eksperymentu 2
GEOMETRIES = ["rect_64", "sunflower_64"]
TRAJECTORIES = ["circle"]  # W tym eksperymencie skupialiśmy się na circle
GAMMAS = [-1, 1, 4, 10, 50, 300]

print(f"--- EKSTRAKCJA KLATEK DO ANALIZY GAMMA ---")
print(f"Czas: {TARGET_TIME}s (Klatka #{FRAME_NO})")
print(f"Geometrie: {GEOMETRIES}")
print(f"Gammy: {GAMMAS}")
print("-" * 50)

count = 0
for geo in GEOMETRIES:
    for traj in TRAJECTORIES:
        for g in GAMMAS:

            # Budowanie ścieżki do pliku wideo
            # Struktura: results/<geo>/<geo>_<traj>/g<gamma>/maps/<geo>_<traj>_g<gamma>.mp4
            folder_name = f"{geo}_{traj}"
            gamma_folder = f"g{g}"
            video_name = f"{geo}_{traj}_g{g}.mp4"

            video_path = RESULTS_DIR / geo / folder_name / gamma_folder / "maps" / video_name

            if not video_path.exists():
                print(f"[SKIP] Nie znaleziono pliku: {video_path}")
                continue

            # Otwarcie wideo
            cap = cv2.VideoCapture(str(video_path))

            # Ustawienie na konkretną klatkę
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NO)
            success, frame = cap.read()

            if success:
                # Nazwa pliku wyjściowego: np. rect_64_circle_g10.png
                output_filename = f"{geo}_{traj}_g{g}.png"
                output_path = OUTPUT_DIR / output_filename

                cv2.imwrite(str(output_path), frame)
                print(f"[OK] Zapisano: {output_filename}")
                count += 1
            else:
                print(f"[ERROR] Nie udało się odczytać klatki z {video_name}")

            cap.release()

print("-" * 50)
print(f"Zakończono! Wyciągnięto {count} obrazów.")
print(f"Pliki gotowe do raportu znajdują się w: {OUTPUT_DIR.resolve()}")