import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class WaveEncoder():
    def __init__(self):
        self.signal = None
        self.time = None
        self.rate = None
        self.period = None
        self.components = None
        self.fs = None
        self.es = None
        self.idxs = None

    def fit(self, signal, s_rate, s_period, s_time, threshold=None, plot=True):
        self.signal = signal
        self.rate = s_rate
        self.period = s_period
        self.time = s_time
        self.components = self.decompose(signal, s_period, s_rate, s_time, threshold=threshold)
        if plot:
            self.plot_fit()
        return self

    def transform(self, signal, positive_only=True):
        """
        Returns frequencies and amplitudes of transformation domain and range
        """
        N = len(signal)
        e = np.fft.fft(signal) / N
        e = np.abs(e)
        f = np.fft.fftfreq(N, self.period)

        if positive_only:
            e = e[range(int(N / 2))]
            f = f[range(int(N / 2))]

        return e, f

    def get_waves(self, amps, freqs, period, rate, time):
        """
        Generate sine waves with given frequency and amplitude.
        """
        N = rate * time
        t_vec = np.arange(N) * period

        waves = []
        for f, a in zip(freqs, amps):
            waves.append(a * np.sin(2 * np.pi * f * t_vec))

        return waves

    def decompose(self, signal, s_period, s_rate, s_time, threshold=None):
        """
        Decompose and return the individual components of a composite wave form.
        Plot each component wave.
        """
        es, fs = self.transform(signal, s_period)
        self.es, self.fs = es, fs

        self.idxs, _ = find_peaks(es, threshold=threshold)
        amps, freqs = es[self.idxs], fs[self.idxs]

        return self.get_waves(amps, freqs, s_period, s_rate, s_time)

    def plot_original(self):
        try:
            N = self.rate * self.time
            t_vec = np.arange(N) * self.period
            plt.plot(t_vec, self.signal, '.')
            plt.title('Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except ValueError:
            print('plt_original: Encoder must be fit against composite wave before plotting')
            raise

    def plot_components(self):
        try:
            N = self.rate * self.time
            t_vec = np.arange(N) * self.period

            fig, axes = plt.subplots(len(self.components), 1)
            for i, wave in enumerate(self.components):
                axes[i].plot(t_vec, wave)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except ValueError:
            print('plt_components: Encoder must be fit against composite wave before plotting')
            raise

    def plot_energy(self):
        try:
            plt.plot(self.fs, self.es, 'b.--', label='Energy vs Frequency')
            plt.plot(self.fs[self.idxs],
                     self.es[self.idxs],
                     'ro', label=f'Peak Frequencies:\n{self.fs[self.idxs]}')
            plt.xlabel('Frequency')
            plt.ylabel('Frequency Strength')
            plt.gca().set_xscale('log')
            plt.legend(), plt.grid()
            plt.show()
        except ValueError:
            print('plot_energy: Encoder must be fit against composite wave before plotting')
            raise

    def plot_fit(self):
            self.plot_original()
            self.plot_energy()
            self.plot_components()

if __name__ == '__main__':
    f1 = 1
    f2 = .5

    # Sample rate, sample period, time to sample
    Fs = 100
    T = 1/Fs
    t = 10

    # 'Sample' signal
    N = Fs * t
    t_vec = np.arange(N) * T
    signal = np.sin(2 * np.pi * f1 * t_vec) + np.sin(2 * np.pi * f2 * t_vec)

    encoder = WaveEncoder().fit(signal, Fs, T, t, threshold=0.12)