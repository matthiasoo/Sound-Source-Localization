import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from pathlib import Path

GEOM_FILE = "geom_3d/sphere_fib_64.xml"
Z_START = 5.0
Z_END = -5.0
RADIUS = 4.0
TURNS = 2.0
DURATION = 15.0


def load_mics_from_xml(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    mics = []
    for pos in root.findall("pos"):
        x = float(pos.get("x"))
        y = float(pos.get("y"))
        z = float(pos.get("z"))
        mics.append([x, y, z])

    return np.array(mics)


def get_helix_points():
    times = np.arange(0, DURATION, 0.05)
    z_values = np.linspace(Z_START, Z_END, len(times))
    total_angle = 2 * np.pi * TURNS
    angles = (times / DURATION) * total_angle

    xs = RADIUS * np.cos(angles)
    ys = RADIUS * np.sin(angles)
    zs = z_values

    return xs, ys, zs


def plot_scenario(mics, traj_x, traj_y, traj_z):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(mics[:, 0], mics[:, 1], mics[:, 2],
               c='blue', marker='o', s=50, depthshade=False, label=f'Microphones ({len(mics)})')

    if "cube" in GEOM_FILE:
        for i in range(len(mics)):
            for j in range(i + 1, len(mics)):
                dist = np.linalg.norm(mics[i] - mics[j])
                if dist < 1.1 and dist > 0.9:
                    ax.plot([mics[i, 0], mics[j, 0]], [mics[i, 1], mics[j, 1]], [mics[i, 2], mics[j, 2]], 'b-',
                            alpha=0.2)

    ax.plot(traj_x, traj_y, traj_z, c='red', linestyle='--', linewidth=2, label='Drone Path (Helix)')

    ax.scatter(traj_x[0], traj_y[0], traj_z[0], c='green', s=100, marker='^', label='Start')
    ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], c='red', s=100, marker='v', label='End')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Scenario Visualization\nGeometry: {GEOM_FILE} | Trajectory: Helix')
    ax.legend()

    all_x = np.concatenate([mics[:, 0], traj_x])
    all_y = np.concatenate([mics[:, 1], traj_y])
    all_z = np.concatenate([mics[:, 2], traj_z])

    max_range = np.array([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]).max() / 2.0
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == "__main__":
    try:
        mics_coords = load_mics_from_xml(GEOM_FILE)
        print(f"Loaded geometry with {len(mics_coords)} microphones.")

        tx, ty, tz = get_helix_points()
        print(f"Generated trajectory with {len(tx)} points.")

        plot_scenario(mics_coords, tx, ty, tz)

    except Exception as e:
        print(f"Błąd: {e}")