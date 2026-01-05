import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- KONFIGURACJA (bez zmian) ---
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis/GAMMA_CHECK")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

geometries = ["rect_64", "sunflower_64"]
gammas_order = [-1, 1, 4, 10, 50, 300]
gamma_labels = ["MVDR (g=-1)", "DAS (g=1)", "g=4", "g=10", "g=50", "g=300"]
trajectory = "circle"

stats = {geo: {'means': [], 'stds': []} for geo in geometries}

print("Generowanie wykresu szerokości wiązki...")

# --- OBLICZENIA (bez zmian) ---
for geom in geometries:
    for g in gammas_order:
        npy_path = RESULTS_DIR / geom / f"{geom}_{trajectory}" / f"g{g}" / "data" / f"{geom}_{trajectory}_widths.npy"

        if npy_path.exists():
            widths = np.load(npy_path)
            avg_frame_width = np.mean(widths, axis=1)
            valid_widths = avg_frame_width[avg_frame_width > 0.01]

            if len(valid_widths) > 0:
                mean_val = np.mean(valid_widths)
                std_val = np.std(valid_widths)
            else:
                mean_val = 0
                std_val = 0
        else:
            mean_val = 0
            std_val = 0

        stats[geom]['means'].append(mean_val)
        stats[geom]['stds'].append(std_val)

# --- RYSOWANIE ---
plt.figure(figsize=(12, 7))  # Nieco szerszy wykres, żeby napisy się mieściły

x = np.arange(len(gammas_order))

styles = {'rect_64': 's--', 'sunflower_64': 'o-'}
colors = {'rect_64': 'blue', 'sunflower_64': 'orange'}

for geom in geometries:
    means = np.array(stats[geom]['means'])
    stds = np.array(stats[geom]['stds'])

    # Maskujemy zera (brak danych)
    mask = means > 0

    # Rysowanie linii i obszaru błędu
    plt.plot(x[mask], means[mask], styles[geom], label=geom, color=colors[geom], linewidth=2, markersize=8)
    plt.fill_between(x[mask], means[mask] - stds[mask], means[mask] + stds[mask], color=colors[geom], alpha=0.15)

    # --- DODAWANIE ETYKIET WARTOŚCI ---
    # Przesunięcie pionowe etykiet, żeby nie nachodziły na linię
    y_offset = 15 if geom == "sunflower_64" else -20

    for xi, yi in zip(x[mask], means[mask]):
        plt.annotate(
            f"{yi:.3f} m",  # Tekst (np. "0.125 m")
            (xi, yi),  # Pozycja punktu
            textcoords="offset points",  # Układ współrzędnych (punkty względem pozycji)
            xytext=(0, y_offset),  # Przesunięcie (0 w bok, y_offset w górę/dół)
            ha='center',  # Wyrównanie w poziomie
            color=colors[geom],  # Kolor tekstu taki sam jak linii
            fontsize=9,
            fontweight='bold'
        )

plt.xticks(x, gamma_labels)
plt.xlabel("Algorytm / Wartość Gamma")
plt.ylabel("Szerokość wiązki głównej (-3dB) [m]")
plt.title("Wpływ parametru Gamma na rozdzielczość przestrzenną")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

save_path = OUTPUT_DIR / "beamwidth_comparison_labels.png"
plt.savefig(save_path, dpi=150)
print(f"Zapisano wykres: {save_path}")