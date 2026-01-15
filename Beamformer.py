import acoular as ac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from pathlib import Path
import time
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

class Beamformer:
    def __init__(self, geom_path, inputfile_path, gamma, algorithm='BeamformerFunctional'):
        ac.config.global_caching = 'none'
        self.algorithm = algorithm
        self.inputfile = Path(inputfile_path)
        self.micgeofile = Path(geom_path)
        self.gamma = gamma

        self.geom_name = self.micgeofile.stem
        self.signal_name = self.inputfile.stem

        self.base_output_path = Path("results") / self.geom_name / self.signal_name / f"g{self.gamma}"

        self.paths = {
            "maps": self.base_output_path / "maps",
            "data": self.base_output_path / "data",
            "times": self.base_output_path / "times"
        }

        self._prepare_directories()

        self.i = 0
        self.frames = []

        print(f"Processing started: {self.inputfile.name} | Algorithm: {self.algorithm}")
        self._run()

    def _prepare_directories(self):
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

    def _map_index_to_range(self, i, num, v_min=0, v_max=1):
        step = (v_max - v_min) / (num - 1)
        return v_min + (i * step)

    def _calc_beam_width(self, map_data, grid, pixel_size, db_drop=3):
        map_db = 10 * np.log10(map_data.T + 1e-12)

        max_val = np.max(map_db)
        limit = max_val - db_drop

        mask = map_db >= limit

        if not np.any(mask):
            return 0.0, 0.0

        x_indices = np.where(np.any(mask, axis=0))[0]
        y_indices = np.where(np.any(mask, axis=1))[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            return 0.0, 0.0

        width_x = (x_indices[-1] - x_indices[0] + 1) * pixel_size
        width_y = (y_indices[-1] - y_indices[0] + 1) * pixel_size

        return width_x, width_y

    def _run(self):
        samplerate, wavData = wavfile.read(self.inputfile)
        ts = ac.TimeSamples(data=wavData, sample_freq=samplerate)
        mg = ac.MicGeom(
            from_file=self.micgeofile)

        rg = ac.RectGrid(
            x_min=-10, x_max=+10,
            y_min=-10, y_max=+10,
            z=10, increment=0.1,
        )
        st = ac.SteeringVector(grid=rg, mics=mg)

        frg_span = 0.2
        FPS = 30
        fbf_gamma = self.gamma

        frames_count = int(ts.num_samples / ts.sample_freq * FPS)
        frame_length = int(ts.sample_freq / FPS)
        print("Frames to be generated: ", frames_count)

        gen = ts.result(frame_length)

        t1 = time.thread_time()
        pt = 0
        total_frame_time = 0
        min_frame_time = 0
        max_frame_time = 0

        self.i = 0
        self.frames = []
        self.widths = []

        for block in gen:
            pt1 = time.thread_time()
            perf_counter_start = time.perf_counter()

            # tempData = block
            tempTS = ac.TimeSamples(data=block, sample_freq=samplerate)
            ps = ac.PowerSpectra(source=tempTS, block_size=128, overlap='50%', window='Hanning', precision='complex128')

            original_csm = ps.csm

            eye_matrix = np.eye(original_csm.shape[1])

            reg_factor = 1e-5

            regularized_csm = original_csm + reg_factor * eye_matrix

            ps_import = ac.PowerSpectraImport(csm=regularized_csm, frequencies=ps.fftfreq())

            # bb = ac.BeamformerFunctional(freq_data=ps, steer=st, gamma=fbf_gamma, r_diag=False)
            bb = ac.BeamformerFunctional(freq_data=ps_import, steer=st, gamma=fbf_gamma, r_diag=False)

            tempRes = np.sum(bb.result[4:32], 0)
            # tempRes = np.nan_to_num(tempRes, nan=0.0)
            r = tempRes.reshape(rg.shape)

            if np.isnan(r).any():
                print("\n!!! WARNING: NaN detected !!!")
            if np.iscomplex(r).any():
                print("\n!!! WARNING: Complex numbers detected !!!")
            # print(f"\nMin: {np.min(r)}, Max: {np.max(r)}")

            p = np.unravel_index(np.argmax(r), r.shape)
            px = self._map_index_to_range(p[0], r.shape[0], rg.extend()[0], rg.extend()[1])
            py = self._map_index_to_range(p[1], r.shape[1], rg.extend()[2], rg.extend()[3])

            pt2 = time.thread_time()
            pt += pt2 - pt1

            INCREMENT = 0.002

            frg = ac.RectGrid(
                x_min=px - frg_span, x_max=px + frg_span,
                y_min=py - frg_span, y_max=py + frg_span,
                z=10, increment=INCREMENT,
            )
            fst = ac.SteeringVector(grid=frg, mics=mg, steer_type='classic')

            bf = ac.BeamformerFunctional(freq_data=ps_import, steer=fst, gamma=fbf_gamma, r_diag=False)

            tempFRes = np.sum(bf.result[8:16], 0)
            # tempFRes = np.nan_to_num(tempFRes, nan=0.0)
            fr = tempFRes.reshape(frg.shape)

            fp = np.unravel_index(np.argmax(fr), fr.shape)
            fpx = self._map_index_to_range(fp[0], fr.shape[0], frg.extend()[0], frg.extend()[1])
            fpy = self._map_index_to_range(fp[1], fr.shape[1], frg.extend()[2], frg.extend()[3])

            wx, wy = self._calc_beam_width(fr, frg, INCREMENT, db_drop=3)
            self.widths.append((wx, wy))

            self.frames.append((r, (px, py), fr, (fpx, fpy)))

            perf_counter_stop = time.perf_counter() - perf_counter_start
            total_frame_time += perf_counter_stop
            max_frame_time = max(max_frame_time, perf_counter_stop)
            if self.i == 0:
                min_frame_time = perf_counter_stop
            min_frame_time = min(min_frame_time, perf_counter_stop)

            print(f"\rBF: {self.i}", end="", flush=True)
            self.i += 1

        print()

        t2 = time.thread_time()
        avg_frame_time = total_frame_time / self.i if self.i > 0 else 0

        print("First stage (low res) time: ", pt, 's')
        print("Second stage (high res) time: ", t2 - t1, 's')
        print("Total frame time: ", total_frame_time, 's')
        print("Average frame time: ", avg_frame_time, 's')
        print("Max frame time: ", max_frame_time, 's')
        print("Min frame time: ", min_frame_time, 's')

        with open(self.paths["times"] / f"times_g{self.gamma}.log", "a") as f:
            f.write(
                f"{self.inputfile.stem},{self.algorithm},{pt},{t2 - t1},{total_frame_time},{avg_frame_time},{max_frame_time},{min_frame_time}\n")

        points = np.array([frame[1] for frame in self.frames])  # (px, py)
        focus_points = np.array([frame[3] for frame in self.frames])  # (fpx, fpy)

        np.save(self.paths["data"] / f"{self.inputfile.stem}_g{self.gamma}_points.npy", points)
        np.save(self.paths["data"] / f"{self.inputfile.stem}_g{self.gamma}_focuspoints.npy", focus_points)

        widths_arr = np.array(self.widths)
        np.save(self.paths["data"] / f"{self.inputfile.stem}_widths.npy", widths_arr)

        self._generate_animation(rg, frg_span, FPS, frames_count)

    def _generate_animation(self, rg, frg_span, FPS, frames_count):
        print("Animation generating...")
        fig, ax = plt.subplots()

        self.anim_i = 0

        def init():
            ax.clear()
            ax.axis("off")

        def update(frame_data):
            self.anim_i += 1
            print(f"\rFrame {self.anim_i}/{frames_count}", end="", flush=True)

            res = frame_data[0]
            p = frame_data[1]
            fres = frame_data[2]
            fp = frame_data[3]

            ax.clear()

            ax.imshow(
                np.transpose(res),
                extent=rg.extend(),
                origin="lower",
            )

            ax.imshow(
                np.transpose(fres),
                extent=(p[0] - frg_span, p[0] + frg_span, p[1] - frg_span, p[1] + frg_span),
                origin="lower",
            )

            ax.plot(fp[0], fp[1], 'r+')

            ax.annotate(f'({fp[0]:0,.2f}, {fp[1]:0,.2f})',
                        xy=(fp[0], fp[1]),
                        xytext=(fp[0] + frg_span, fp[1] + frg_span),
                        color='white')

        ani = animation.FuncAnimation(fig, update, frames=self.frames, init_func=init, repeat=True, interval=1 / FPS)

        output_file = self.paths["maps"] / f"{self.inputfile.stem}_g{self.gamma}.mp4"
        ani.save(output_file, writer="ffmpeg", fps=FPS)
        plt.close()
        print(f"\nAnimation saved: {output_file}")

class Beamformer3D:
    def __init__(self, geom_path, inputfile_path, gamma, algorithm='BeamformerFunctional'):
        ac.config.global_caching = 'none'
        self.algorithm = algorithm
        self.inputfile = Path(inputfile_path)
        self.micgeofile = Path(geom_path)
        self.gamma = gamma

        self.geom_name = self.micgeofile.stem
        self.signal_name = self.inputfile.stem

        self.base_output_path = Path("results_3D") / self.geom_name / self.signal_name / f"g{self.gamma}"

        self.paths = {
            "maps": self.base_output_path / "maps",
            "data": self.base_output_path / "data",
            "times": self.base_output_path / "times"
        }

        self._prepare_directories()

        self.i = 0

        print(f"3D Processing started: {self.inputfile.name} | Algorithm: {self.algorithm}")
        self._run()

    def _prepare_directories(self):
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

    def _map_index_to_range(self, i, num, v_min, v_max):
        if num <= 1: return v_min
        step = (v_max - v_min) / (num - 1)
        return v_min + (i * step)


    def _run(self):
        samplerate, wavData = wavfile.read(self.inputfile)
        ts = ac.TimeSamples(data=wavData, sample_freq=samplerate)
        mg = ac.MicGeom(
            from_file=self.micgeofile)
        INCREMENT = 0.5
        F_INCREMENT = 0.05
        frg_span = 0.4

        rg = ac.RectGrid3D(
            x_min=-6, x_max=+6,
            y_min=-6, y_max=+6,
            z_min=-6, z_max=+6,
            increment=INCREMENT
        )
        st = ac.SteeringVector(grid=rg, mics=mg, steer_type='classic')

        FPS = 30
        fbf_gamma = self.gamma

        frames_count = int(ts.num_samples / ts.sample_freq * FPS)
        frame_length = int(ts.sample_freq / FPS)

        gen = ts.result(frame_length)
        print("Frames to be generated: ", frames_count)
        print(f"Grid size: {rg.size}, {rg.shape}")

        gen = ts.result(frame_length)

        t1 = time.thread_time()
        pt = 0
        total_frame_time = 0
        min_frame_time = 0
        max_frame_time = 0

        self.detected_path = []
        self.i = 0

        for block in gen:
            pt1 = time.thread_time()
            perf_counter_start = time.perf_counter()

            tempTS = ac.TimeSamples(data=block, sample_freq=samplerate)
            ps = ac.PowerSpectra(source=tempTS, block_size=128, overlap='50%', window='Hanning', precision='complex128')

            original_csm = ps.csm
            eye_matrix = np.eye(original_csm.shape[1])
            reg_factor = 1e-5
            regularized_csm = original_csm + reg_factor * eye_matrix
            ps_import = ac.PowerSpectraImport(csm=regularized_csm, frequencies=ps.fftfreq())

            # bb = ac.BeamformerFunctional(freq_data=ps, steer=st, gamma=fbf_gamma, r_diag=False)
            bb = ac.BeamformerFunctional(freq_data=ps_import, steer=st, gamma=fbf_gamma, r_diag=False)

            tempRes = np.sum(bb.result[4:32], 0)
            # tempRes = np.nan_to_num(tempRes, nan=0.0)
            r = tempRes.reshape(rg.shape)

            if np.isnan(r).any():
                print("\n!!! WARNING: NaN detected !!!")
            if np.iscomplex(r).any():
                print("\n!!! WARNING: Complex numbers detected !!!")
            # print(f"\nMin: {np.min(r)}, Max: {np.max(r)}")

            p = np.unravel_index(np.argmax(r), r.shape)
            px = self._map_index_to_range(p[0], r.shape[0], rg.x_min, rg.x_max)
            py = self._map_index_to_range(p[1], r.shape[1], rg.y_min, rg.y_max)
            pz = self._map_index_to_range(p[2], r.shape[2], rg.z_min, rg.z_max)

            pt2 = time.thread_time()
            pt += pt2 - pt1

            frg = ac.RectGrid3D(
                x_min=px - frg_span, x_max=px + frg_span,
                y_min=py - frg_span, y_max=py + frg_span,
                z_min=pz - frg_span, z_max=pz + frg_span,
                increment=F_INCREMENT
            )
            fst = ac.SteeringVector(grid=frg, mics=mg, steer_type='classic')

            bf = ac.BeamformerFunctional(freq_data=ps_import, steer=fst, gamma=fbf_gamma, r_diag=False)

            tempFRes = np.sum(bf.result[8:16], 0)
            # tempFRes = np.nan_to_num(tempFRes, nan=0.0)
            fr = tempFRes.reshape(frg.shape)

            fp = np.unravel_index(np.argmax(fr), fr.shape)
            fpx = self._map_index_to_range(fp[0], fr.shape[0], frg.x_min, frg.x_max)
            fpy = self._map_index_to_range(fp[1], fr.shape[1], frg.y_min, frg.y_max)
            fpz = self._map_index_to_range(fp[2], fr.shape[2], frg.z_min, frg.z_max)

            self.detected_path.append([fpx, fpy, fpz])

            perf_counter_stop = time.perf_counter() - perf_counter_start
            total_frame_time += perf_counter_stop
            max_frame_time = max(max_frame_time, perf_counter_stop)
            if self.i == 0:
                min_frame_time = perf_counter_stop
            min_frame_time = min(min_frame_time, perf_counter_stop)

            print(f"\rFrame {self.i+1} | Pos: ({fpx:.2f}, {fpy:.2f}, {fpz:.2f})", end="", flush=True)
            self.i += 1

        print()

        t2 = time.thread_time()
        avg_frame_time = total_frame_time / self.i if self.i > 0 else 0

        print("First stage (low res) time: ", pt, 's')
        print("Second stage (high res) time: ", t2 - t1, 's')
        print("Total frame time: ", total_frame_time, 's')
        print("Average frame time: ", avg_frame_time, 's')
        print("Max frame time: ", max_frame_time, 's')
        print("Min frame time: ", min_frame_time, 's')

        with open(self.paths["times"] / f"times_g{self.gamma}.log", "a") as f:
            f.write(
                f"{self.inputfile.stem},{self.algorithm},{pt},{t2 - t1},{total_frame_time},{avg_frame_time},{max_frame_time},{min_frame_time}\n")

        path_arr = np.array(self.detected_path)
        np.save(self.paths["data"] / f"{self.inputfile.stem}_g{self.gamma}_path3d.npy", path_arr)

        self._generate_animation_3d(path_arr, FPS)

    def _generate_animation_3d(self, path_data, fps):
        print("Generating 3D Animation...")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        line, = ax.plot([], [], [], 'b-', alpha=0.5, linewidth=1, label='History')
        point, = ax.plot([], [], [], 'ro', markersize=8, label='Drone')

        title = ax.set_title("3D Tracking")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def update(frame_idx):
            print(f"\rAnim Frame {frame_idx}/{len(path_data)}", end="")
            curr_x = path_data[:frame_idx + 1, 0]
            curr_y = path_data[:frame_idx + 1, 1]
            curr_z = path_data[:frame_idx + 1, 2]

            line.set_data(curr_x, curr_y)
            line.set_3d_properties(curr_z)

            if frame_idx < len(path_data):
                px, py, pz = path_data[frame_idx]
                point.set_data([px], [py])
                point.set_3d_properties([pz])
                title.set_text(f"Pos: {px:.1f}, {py:.1f}, {pz:.1f}")

            return line, point

        ani = animation.FuncAnimation(fig, update, frames=len(path_data), init_func=init, interval=1000 / fps)

        output_file = self.paths["maps"] / f"{self.inputfile.stem}_g{self.gamma}_3d.mp4"
        ani.save(output_file, writer="ffmpeg", fps=fps)
        plt.close()
        print(f"\nAnimation saved: {output_file}")