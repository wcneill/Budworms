import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class WaveEncoder():
    def __init__(self):
        self.wave = None
        self.time = None
        self.components = None
        self.wind_freqs = None
        self.comp_freqs = None
        self.comp_energ = None
        self.energies   = None

    def fit(self, wave, time_series, freqs, threshold=None, plot=True):
        self.wave = wave
        self.time = time_series
        self.wind_freqs = freqs
        self.components = self.decompose(time_series, wave, freqs, threshold=threshold)
        if plot:
            self.plot_fit()
        return self

    def wind(self, timescale, data, w_freq):
        """
        wrap time-series data around a circle on complex plain
        at given winding frequency.
        """
        return data * np.exp(2 * np.pi * w_freq * timescale * 1.j)

    def transform(self, x, y, freqs):
        """
        Returns center of mass of each winding frequency
        """
        ft = []
        for f in freqs:
            mapped = self.wind(x, y, f)
            re, im = np.real(mapped).mean(), np.imag(mapped).mean()
            mag = np.sqrt(re ** 2 + im ** 2)
            ft.append(mag)

        return np.array(ft)

    def get_waves(self, parts, time):
        """
        Generate sine waves based on frequency parts.
        """
        num_waves = len(parts)
        steps = len(time)
        waves = np.zeros((num_waves, steps))
        for i in range(num_waves):
            waves[i] = np.sin(parts[i] * 2 * np.pi * time)

        return waves


    def decompose(self, time, data, freqs, threshold=None):
        """
        Decompose and return the individual components of a composite wave form.
        Plot each component wave.
        """
        self.energies = self.transform(time, data, freqs)
        peaks, _ = find_peaks(self.energies, threshold=threshold)
        self.comp_freqs = freqs[peaks]
        self.comp_energ = self.energies[peaks]

        return self.get_waves(self.comp_freqs, time)

    def plot_original(self):
        try:
            plt.plot(self.time, self.wave, '.')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except ValueError:
            print('Encoder must be fit against composite wave before plotting')
            exit(1)


    def plot_components(self):
        try:
            for wave, freq in zip(self.components, self.comp_freqs):
                plt.plot(self.time, wave, label=f'Frequency: {freq} Hz')
            plt.legend()
            plt.show()
        except ValueError:
            print('Encoder must be fit against composite wave before plotting')
            exit(1)

    def plot_energy(self):
        try:
            plt.plot(self.wind_freqs, self.energies, 'b.--', label='Energy vs Frequency')
            plt.plot(self.comp_freqs, self.comp_energ, 'ro', label='Peaks')
            plt.xlabel('Frequency')
            plt.legend(), plt.grid()
            plt.show()
        except ValueError:
            print('Encoder must be fit against composite wave before plotting')
            exit(1)

    def plot_fit(self):
            self.plot_original()
            self.plot_energy()
            self.plot_components()

if __name__ == '__main__':
    f1 = 3
    f2 = 5
    x = np.linspace(0, 1, 200)
    y = np.cos(f1 * 2 * np.pi * x) + np.sin(f2 * 2 * np.pi * x)
    winding_freqs = np.arange(0, 20, .5)
    encoder = WaveEncoder().fit(y, x, winding_freqs, threshold=0.12)

