import copy
import os
import acoular as ac
import numpy as np
from IPython.core.pylabtools import figsize
from acoular.internal import digest
from traits.api import cached_property, Int, List, Property
from pathlib import Path
import matplotlib.pyplot as plt

ac.config.global_caching = 'none'

class DroneSignalGenerator(ac.NoiseGenerator):
    # defaults
    rpm_list = List([15000, ])
    num_blades_per_rotor = Int(2)

    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'rpm_list', 'num_blades_per_rotor'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        # initialize a random generator for noise generation
        rng = np.random.default_rng(seed=self.seed)
        # use 1/f² broadband noise as basis for the signal
        wn = rng.standard_normal(self.num_samples)  # WHITE NOISE
        wnf = np.fft.rfft(wn)  # to freq domain
        wnf /= (np.linspace(0.1, 1, len(wnf)) * 5) ** 2  # RED NOISE
        sig = np.fft.irfft(wnf)  # to time domain

        # vector with all time instances
        t = np.arange(self.num_samples, dtype=float) / self.sample_freq

        # iterate over all rotors
        for rpm in self.rpm_list:
            f_base = rpm / 60  # rotor speed in Hz

            # randomly set phase of rotor
            phase = rng.uniform() * 2 * np.pi

            # calculate higher harmonics up to 50 times the rotor speed
            for n in np.arange(50) + 1:
                # if we're looking at a blade passing frequency, make it louder
                if n % self.num_blades_per_rotor == 0:
                    amp = 1
                else:
                    amp = 0.2

                # exponentially decrease amplitude for higher freqs with arbitrary factor
                amp *= np.exp(-n / 10)

                # add harmonic signal component to existing signal
                sig += amp * np.sin(2 * np.pi * n * f_base * t + phase)

                # return signal normalized to given RMS value
        return sig * self.rms / np.std(sig)


class SignalRecorder:
    def __init__(self, geom_path, signal_src):
        self.geom_path = Path(geom_path)
        if not self.geom_path.exists():
            raise FileNotFoundError(f"Geom file not found: {geom_path}")
        self.mics = ac.MicGeom(from_file=geom_path)
        self.signal = signal_src
        self.env = ac.Environment()
        self.output_dir = Path('signal/dynamic')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name = self.geom_path.stem

        self.duration = self.signal.num_samples / self.signal.sample_freq

    def _generate_and_save(self, trajectory, suffix):

        # main src
        p = ac.MovingPointSourceDipole(
            signal=self.signal,
            trajectory=trajectory,
            mics=self.mics,
            env=self.env,
            conv_amp=True,
            start=0.0,
            direction=(0, 0, 1)
        )

        traj_points = trajectory.points
        mirror_points = {t: (coord[0], coord[1], -coord[2]) for t, coord in traj_points.items()}
        traj_mirror = ac.Trajectory(points=mirror_points)

        # mirror src
        p_reflection = ac.MovingPointSourceDipole(
            signal=self.signal,
            trajectory=traj_mirror,
            conv_amp=True,
            mics=self.mics,
            start=0.0,
            env=self.env,
            direction=(0, 0, -1)
        )

        # noise
        wn_gen = ac.WNoiseGenerator(
            sample_freq=self.signal.sample_freq,
            num_samples=self.signal.num_samples,
            seed=100,
            # rms=0.05
            rms=0.8
        )
        n = ac.UncorrelatedNoiseSource(
            signal=wn_gen,
            mics=self.mics
        )

        drone_mix = ac.SourceMixer(sources=[p, p_reflection, n])

        file_wav = self.output_dir / f"{self.output_name}_{suffix}.wav"
        all_channels = list(range(self.mics.num_mics))

        if file_wav.exists():
            try:
                os.remove(file_wav)
            except PermissionError:
                print(f"Cannot delete {file_wav}. Is it open?")

        output = ac.WriteWAV(source=drone_mix, name=str(file_wav), channels=all_channels)
        output.save()
        print(f"--> Saved: {file_wav}")

    def run_linear(self, height=10.0):
        print(f"Generating LINEAR path for {self.output_name}...")

        start_pos = -9.0
        end_pos = 9.0

        times = np.arange(0, self.duration, 0.1)
        velocity = (end_pos - start_pos) / self.duration

        waypoints = {}
        for t in times:
            x = start_pos + velocity * t
            waypoints[t] = (x, 0.0, height)

        traj = ac.Trajectory(points=waypoints)
        self._generate_and_save(traj, "linear")

    def run_diagonal(self, height=10.0):
        print(f"Generating DIAGONAL path for {self.output_name}...")

        start_x, start_y = -8.0, -8.0
        end_x, end_y = 8.0, 8.0

        times = np.arange(0, self.duration, 0.1)

        waypoints = {}
        for t in times:
            progress = t / self.duration  # 0.0 do 1.0
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            waypoints[t] = (x, y, height)

        traj = ac.Trajectory(points=waypoints)
        self._generate_and_save(traj, "diagonal")

    def run_circle(self, radius=8.0, height=10.0):
        print(f"Generating CIRCLE path for {self.output_name}...")

        times = np.arange(0, self.duration, 0.05)

        omega = 2 * np.pi / self.duration

        waypoints = {}
        for t in times:
            angle = omega * t + np.pi

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            waypoints[t] = (x, y, height)

        traj = ac.Trajectory(points=waypoints)
        self._generate_and_save(traj, "circle")