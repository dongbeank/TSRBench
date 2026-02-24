import numpy as np
from ads_evt.spot import SPOT, biSPOT, dSPOT, bidSPOT
import os
import pandas as pd
import random
import time
from sklearn.preprocessing import StandardScaler

class CollectiveNoise:
    def __init__(self, seed=2025, level_shift_args=None, exp_spike_args=None,
                 gaussian_args=None, spot_args=None, min_sigma=0.01):
        """
        Initialize the CollectiveNoise class with default parameters for all noise types.

        Parameters:
        - seed: Random seed for reproducibility
        - level_shift_args: Dict {level: {freq, dur, amp}} for level shift
        - exp_spike_args: Dict {level: {freq, dur, amp}} for exponential spike
        - gaussian_args: Dict {level: {sigma}} for Gaussian noise
        - spot_args: Dict for SPOT algorithm configuration
        - min_sigma: Minimum sigma floor for Gaussian noise (prevents zero noise on constant columns)
        """
        if level_shift_args is None:
            self.level_shift_args_map = {
                1: {'freq': 0.002, 'dur': 6, 'amp': 0.0016},
                2: {'freq': 0.004, 'dur': 9, 'amp': 0.0016},
                3: {'freq': 0.004, 'dur': 12, 'amp': 0.0004},
                4: {'freq': 0.008, 'dur': 12, 'amp': 0.0004},
                5: {'freq': 0.008, 'dur': 15, 'amp': 0.0001},
            }
        else:
            self.level_shift_args_map = level_shift_args

        if exp_spike_args is None:
            self.exp_spike_args_map = {
                1: {'freq': 0.002, 'dur': 6, 'amp': 0.0016},
                2: {'freq': 0.004, 'dur': 9, 'amp': 0.0016},
                3: {'freq': 0.004, 'dur': 12, 'amp': 0.0004},
                4: {'freq': 0.008, 'dur': 12, 'amp': 0.0004},
                5: {'freq': 0.008, 'dur': 15, 'amp': 0.0001},
            }
        else:
            self.exp_spike_args_map = exp_spike_args

        if gaussian_args is None:
            self.gaussian_args_map = {
                1: {'sigma': 0.1},
                2: {'sigma': 0.2},
                3: {'sigma': 0.3},
                4: {'sigma': 0.5},
                5: {'sigma': 0.8},
            }
        else:
            self.gaussian_args_map = gaussian_args

        self.min_sigma = min_sigma

        if spot_args is None:
            self.spot_args = {
                'type': 'bidspot',
                'n_points': 8,
                'depth': 0.01,
                'init_points': 0.05,
                'init_level': 0.98
            }
        else:
            self.spot_args = spot_args

        self.evt_values = {}
        self.evt_for_col = []

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------ #
    #  Level Shift
    # ------------------------------------------------------------------ #

    def custom_inject_level_shift(self, X: np.array, freq, dur, amp, exclude_mask=None):
        """
        Injects a level shift into the input signal X.

        Parameters:
        - X: Input signal (1D numpy array)
        - freq: Frequency parameter for Poisson process (anomaly rate)
        - dur: Duration parameter for geometric distribution
        - amp: Amplitude parameter (SPOT risk level q)
        - exclude_mask: Boolean array; True at indices that must not receive noise

        Returns:
        - noise: Injected level shift noise array
        """
        if X.ndim != 1:
            raise ValueError("Input signal X must be a 1D numpy array.")

        noise = np.zeros_like(X)

        T = 2 * X.shape[0] - 1
        N = np.random.poisson(freq * T)
        init_data = min(5000, int(self.spot_args['init_points'] * X.shape[0]))

        m = np.random.randint(0, T + 1, N)
        m = m[m >= X.shape[0]]
        m = m - X.shape[0]
        n = m.shape[0]

        dur = 1 / (dur - 1)
        d = np.random.geometric(dur, n) + 1

        # Filter out anomalies that overlap with excluded positions
        if exclude_mask is not None and n > 0:
            valid = []
            for i in range(n):
                end = min(m[i] + d[i], X.shape[0])
                if not np.any(exclude_mask[m[i]:end]):
                    valid.append(i)
            if valid:
                m = m[np.array(valid)]
                d = d[np.array(valid)]
                n = len(valid)
            else:
                return noise

        if n == 0:
            return noise

        try:
            if self.spot_args['type'] == 'spot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = SPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bispot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = biSPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            elif self.spot_args['type'] == 'dspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = dSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bidspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = bidSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            else:
                raise ValueError("Invalid type for amplitude arguments. Choose from 'spot', 'bispot', 'dspot', or 'bidspot'.")

        except ValueError as e:
            print(f"Error in amplitude arguments: {e}")
            raise

        for i in range(n):
            if self.spot_args['type'] == 'spot' or self.spot_args['type'] == 'dspot':
                noise[m[i]:m[i] + d[i]] += np.min(upper[m[i]:m[i] + d[i]])
            else:
                if np.random.rand() < 0.5:
                    noise[m[i]:min(m[i] + d[i], X.shape[0])] = np.maximum(np.min(upper[m[i]:min(m[i] + d[i], X.shape[0])]), noise[m[i]:min(m[i] + d[i], X.shape[0])])
                else:
                    noise[m[i]:min(m[i] + d[i], X.shape[0])] = np.minimum(np.max(lower[m[i]:min(m[i] + d[i], X.shape[0])]), noise[m[i]:min(m[i] + d[i], X.shape[0])])

        return noise

    def inject_level_shift(self, X: np.array, noise_level: int, exclude_mask=None):
        """
        Injects level shift noise using predefined parameters for the specified noise level.
        """
        return self.custom_inject_level_shift(
            X,
            self.level_shift_args_map[noise_level]['freq'],
            self.level_shift_args_map[noise_level]['dur'],
            self.level_shift_args_map[noise_level]['amp'],
            exclude_mask=exclude_mask,
        )

    # ------------------------------------------------------------------ #
    #  Exponential Spike
    # ------------------------------------------------------------------ #

    def custom_inject_exp_spike(self, X: np.array, freq, dur, amp, exclude_mask=None):
        """
        Injects an exponential spike into the input signal X.

        Parameters:
        - X: Input signal (1D numpy array)
        - freq: Frequency parameter for Poisson process
        - dur: Duration parameter for geometric distribution
        - amp: Amplitude parameter (SPOT risk level q)
        - exclude_mask: Boolean array; True at indices that must not receive noise

        Returns:
        - noise: Injected exponential spike noise array
        """
        if X.ndim != 1:
            raise ValueError("Input signal X must be a 1D numpy array.")

        noise = np.zeros_like(X)

        T = 2 * X.shape[0] - 1
        N = np.random.poisson(freq * T)
        init_data = min(int(self.spot_args['init_points'] * X.shape[0]), 5000)

        m = np.random.randint(0, T + 1, N)
        m = m[m >= X.shape[0]]
        m = m - X.shape[0]
        n = m.shape[0]

        dur = 2 / dur
        d1 = np.random.geometric(dur, n)
        d2 = np.random.geometric(dur, n)

        # Filter out anomalies that overlap with excluded positions
        if exclude_mask is not None and n > 0:
            valid = []
            for i in range(n):
                end = min(m[i] + d1[i] + d2[i] + 1, X.shape[0])
                if not np.any(exclude_mask[m[i]:end]):
                    valid.append(i)
            if valid:
                valid = np.array(valid)
                m = m[valid]
                d1 = d1[valid]
                d2 = d2[valid]
                n = len(valid)
            else:
                return noise

        if n == 0:
            return noise

        try:
            if self.spot_args['type'] == 'spot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = SPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bispot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = biSPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            elif self.spot_args['type'] == 'dspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = dSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bidspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = bidSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            else:
                raise ValueError("Invalid type for amplitude arguments. Choose from 'spot', 'bispot', 'dspot', or 'bidspot'.")

        except ValueError as e:
            print(f"Error in amplitude arguments: {e}")
            raise

        for i in range(n):
            if self.spot_args['type'] == 'spot' or self.spot_args['type'] == 'dspot':
                noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])] += self.__exp_spike_curve(upper[min(m[i] + d1[i], X.shape[0] - 1)], 1e-4, d1[i], d2[i])[:min(d1[i] + d2[i] + 1, X.shape[0] - m[i])]

            else:
                if np.random.rand() < 0.5:
                    noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])] = np.maximum(self.__exp_spike_curve(upper[min(m[i] + d1[i], X.shape[0] - 1)], 1e-4, d1[i], d2[i])[:min(d1[i] + d2[i] + 1, X.shape[0] - m[i])], noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])])
                else:
                    noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])] = np.minimum(self.__exp_spike_curve(lower[min(m[i] + d1[i], X.shape[0] - 1)], 1e-4, d1[i], d2[i])[:min(d1[i] + d2[i] + 1, X.shape[0] - m[i])], noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])])

        return noise

    def inject_exp_spike(self, X: np.array, noise_level: int, exclude_mask=None):
        """
        Injects exponential spike noise using predefined parameters for the specified noise level.
        """
        return self.custom_inject_exp_spike(
            X,
            self.exp_spike_args_map[noise_level]['freq'],
            self.exp_spike_args_map[noise_level]['dur'],
            self.exp_spike_args_map[noise_level]['amp'],
            exclude_mask=exclude_mask,
        )

    def __exp_spike_curve(self, h, beta, epsilon1, epsilon2):
        """
        Generates an exponential spike curve function.
        """
        x = np.arange(0, epsilon1 + epsilon2 + 1, dtype=float)
        y = np.zeros_like(x)

        y[:epsilon1] = h / np.exp(np.log(beta) / epsilon1 * (x[:epsilon1] - epsilon1))
        y[epsilon1:] = h * np.exp(np.log(beta) / epsilon2 * (x[epsilon1:] - epsilon1))
        return y

    # ------------------------------------------------------------------ #
    #  Impulse Spike (single-point)
    # ------------------------------------------------------------------ #

    def custom_inject_impulse(self, X: np.array, freq, amp, exclude_mask=None):
        """
        Injects single-point impulse spikes with EVT-calibrated amplitude.

        Parameters:
        - X: Input signal (1D numpy array)
        - freq: Frequency parameter for Poisson process
        - amp: Amplitude parameter (SPOT risk level q)
        - exclude_mask: Boolean array; True at indices that must not receive noise

        Returns:
        - noise: Injected impulse noise array
        """
        if X.ndim != 1:
            raise ValueError("Input signal X must be a 1D numpy array.")

        noise = np.zeros_like(X)

        T = 2 * X.shape[0] - 1
        N = np.random.poisson(freq * T)
        init_data = min(int(self.spot_args['init_points'] * X.shape[0]), 5000)

        m = np.random.randint(0, T + 1, N)
        m = m[m >= X.shape[0]]
        m = m - X.shape[0]
        n = m.shape[0]

        # Filter excluded positions
        if exclude_mask is not None and n > 0:
            valid = np.array([i for i in range(n) if not exclude_mask[m[i]]])
            if len(valid) > 0:
                m = m[valid]
                n = len(m)
            else:
                return noise

        if n == 0:
            return noise

        try:
            if self.spot_args['type'] == 'spot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = SPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bispot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = biSPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            elif self.spot_args['type'] == 'dspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = dSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bidspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = bidSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            else:
                raise ValueError("Invalid type for amplitude arguments. Choose from 'spot', 'bispot', 'dspot', or 'bidspot'.")

        except ValueError as e:
            print(f"Error in amplitude arguments: {e}")
            raise

        for i in range(n):
            if self.spot_args['type'] == 'spot' or self.spot_args['type'] == 'dspot':
                noise[m[i]] += upper[m[i]]
            else:
                if np.random.rand() < 0.5:
                    noise[m[i]] = max(upper[m[i]], noise[m[i]])
                else:
                    noise[m[i]] = min(lower[m[i]], noise[m[i]])

        return noise

    def inject_impulse(self, X: np.array, noise_level: int, exclude_mask=None):
        """
        Injects impulse (single-point spike) noise using predefined parameters.
        Uses the same freq/amp as exponential spike.
        """
        return self.custom_inject_impulse(
            X,
            self.exp_spike_args_map[noise_level]['freq'],
            self.exp_spike_args_map[noise_level]['amp'],
            exclude_mask=exclude_mask,
        )

    # ------------------------------------------------------------------ #
    #  Gaussian Noise
    # ------------------------------------------------------------------ #

    def inject_gaussian(self, X: np.array, noise_level: int):
        """
        Injects additive Gaussian noise into the input signal (applied to every time step).

        Parameters:
        - X: Input signal (1D numpy array, standardized)
        - noise_level: Noise level determining sigma

        Returns:
        - noise: Gaussian noise array
        """
        sigma = max(self.gaussian_args_map[noise_level]['sigma'], self.min_sigma)
        return np.random.normal(0, sigma, X.shape)

    # ------------------------------------------------------------------ #
    #  Missing (zero-valued, independent positions)
    # ------------------------------------------------------------------ #

    def inject_missing(self, X: np.array, noise_level: int, exclude_mask=None):
        """
        Generate a boolean mask for missing-value positions.

        Uses the same Poisson + Geometric mechanism as level shift (same freq/dur
        parameters) but generates its own independent positions. At masked positions,
        values will be set to zero in original scale.

        Parameters:
        - X: Input signal (1D numpy array)
        - noise_level: Noise level (1-5)
        - exclude_mask: Boolean array; True at indices that must not be masked

        Returns:
        - mask: Boolean array; True at positions that should become zero
        """
        if X.ndim != 1:
            raise ValueError("Input signal X must be a 1D numpy array.")

        freq = self.level_shift_args_map[noise_level]['freq']
        dur = self.level_shift_args_map[noise_level]['dur']

        T = 2 * X.shape[0] - 1
        N = np.random.poisson(freq * T)

        m = np.random.randint(0, T + 1, N)
        m = m[m >= X.shape[0]]
        m = m - X.shape[0]
        n = m.shape[0]

        dur_p = 1 / (dur - 1)
        d = np.random.geometric(dur_p, n) + 1

        # Filter excluded positions
        if exclude_mask is not None and n > 0:
            valid = []
            for i in range(n):
                end = min(m[i] + d[i], X.shape[0])
                if not np.any(exclude_mask[m[i]:end]):
                    valid.append(i)
            if valid:
                m = m[np.array(valid)]
                d = d[np.array(valid)]
                n = len(valid)
            else:
                return np.zeros(X.shape[0], dtype=bool)

        mask = np.zeros(X.shape[0], dtype=bool)
        for i in range(n):
            mask[m[i]:min(m[i] + d[i], X.shape[0])] = True
        return mask

    # ------------------------------------------------------------------ #
    #  Coordinated injection (non-overlapping positions)
    # ------------------------------------------------------------------ #

    def inject_noise(self, X: np.array, noise_level: int):
        """
        Injects both level shift and exponential spike noise into the input signal.
        (Legacy method for backward compatibility.)

        Returns:
        - noise_shift: Level shift noise
        - noise_spike: Exponential spike noise
        """
        noise_shift = self.inject_level_shift(X, noise_level)
        noise_spike = self.inject_exp_spike(X, noise_level)
        return noise_shift, noise_spike

    def inject_all_noise(self, X: np.array, noise_level: int):
        """
        Generate all noise types with non-overlapping anomaly positions.

        Priority order: Shift -> Spike -> Impulse -> Missing -> Gaussian.
        Each type avoids positions already claimed by higher-priority types.

        Parameters:
        - X: Input signal (1D numpy array, standardized)
        - noise_level: Noise level (1-5)

        Returns:
        - dict with keys: 'shift', 'spike', 'impulse', 'gaussian', 'missing_mask'
        """
        # 1. Level shift (first priority)
        shift = self.inject_level_shift(X, noise_level)
        shift_mask = (shift != 0)

        # 2. Exponential spike (avoids shift positions)
        spike = self.inject_exp_spike(X, noise_level, exclude_mask=shift_mask)
        spike_mask = (spike != 0)

        # 3. Impulse (avoids shift + spike positions)
        occupied = shift_mask | spike_mask
        impulse = self.inject_impulse(X, noise_level, exclude_mask=occupied)
        impulse_mask = (impulse != 0)

        # 4. Missing (avoids shift + spike + impulse; independent positions)
        occupied = occupied | impulse_mask
        missing_mask = self.inject_missing(X, noise_level, exclude_mask=occupied)

        # 5. Gaussian (everywhere, independent)
        gaussian = self.inject_gaussian(X, noise_level)

        return {
            'shift': shift,
            'spike': spike,
            'impulse': impulse,
            'gaussian': gaussian,
            'missing_mask': missing_mask,
        }

    # ------------------------------------------------------------------ #
    #  High-level API: corrupt DataFrame / 2D array
    # ------------------------------------------------------------------ #

    def corrupt(self, data, noise_level=3, skip_first_col=True, zero_clip=False):
        """
        Corrupt a DataFrame or 2D numpy array directly.

        Handles standardization, noise injection, and inverse-transformation
        internally. Returns a dict of corrupted results in the same format
        as the input.

        Parameters:
        - data: pandas DataFrame or 2D numpy array (or 1D array for a single series)
        - noise_level: Severity level (1-5)
        - skip_first_col: If True and data is a DataFrame, treat the first column
                          as a non-numeric index (e.g. date) and skip it
        - zero_clip: If True, clip negative values to zero after injection

        Returns:
        - dict with keys: 'shift', 'spike', 'impulse', 'gaussian', 'missing', 'combined'
          Values are in the same format as the input (DataFrame or numpy array).
        """
        is_df = isinstance(data, pd.DataFrame)
        was_1d = False

        if is_df:
            df = data.copy()
            if skip_first_col:
                meta_col = df.iloc[:, 0].copy()
                X_raw = df.iloc[:, 1:].values.copy()
                data_cols = df.columns[1:]
            else:
                meta_col = None
                X_raw = df.values.copy()
                data_cols = df.columns
        else:
            data_np = np.asarray(data, dtype=float)
            if data_np.ndim == 1:
                was_1d = True
                X_raw = data_np.reshape(-1, 1)
            else:
                X_raw = data_np.copy()

        n_cols = X_raw.shape[1]

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Prepare EVT caches
        self.evt_for_col = [{} for _ in range(n_cols)]

        # Output buffers
        out_shift = np.zeros_like(X)
        out_spike = np.zeros_like(X)
        out_impulse = np.zeros_like(X)
        out_gaussian = np.zeros_like(X)
        out_combined = np.zeros_like(X)
        out_missing = X.copy()
        missing_masks = np.zeros(X.shape, dtype=bool)

        np.random.seed(self.seed)
        random.seed(self.seed)

        for i in range(n_cols):
            self.evt_values = self.evt_for_col[i]
            col = X[:, i]

            results = self.inject_all_noise(col, noise_level)

            out_shift[:, i] = results['shift'] + col
            out_spike[:, i] = results['spike'] + col
            out_impulse[:, i] = results['impulse'] + col
            out_gaussian[:, i] = results['gaussian'] + col
            out_missing[:, i] = col
            missing_masks[:, i] = results['missing_mask']

            combined = results['shift'] + results['spike'] + results['impulse']
            out_combined[:, i] = combined + col

            if zero_clip:
                for arr in [out_shift, out_spike, out_impulse, out_gaussian, out_combined]:
                    arr[:, i] = np.clip(arr[:, i], 0, None)

            self.evt_for_col[i] = self.evt_values

        # Inverse transform
        inv = {}
        for key, arr in [('shift', out_shift), ('spike', out_spike),
                         ('impulse', out_impulse), ('gaussian', out_gaussian),
                         ('combined', out_combined)]:
            inv[key] = scaler.inverse_transform(arr)

        inv['missing'] = scaler.inverse_transform(out_missing)
        inv['missing'][missing_masks] = 0
        if zero_clip:
            inv['missing'] = np.clip(inv['missing'], 0, None)

        # Format output
        output = {}
        if is_df:
            for key, arr in inv.items():
                out_df = data.copy()
                out_df[data_cols] = arr
                output[key] = out_df
        elif was_1d:
            for key, arr in inv.items():
                output[key] = arr.ravel()
        else:
            output = inv

        return output

    # ------------------------------------------------------------------ #
    #  CSV dataset generation (batch)
    # ------------------------------------------------------------------ #

    def make_noise_datasets(self, args):
        """
        Creates noisy datasets by injecting all noise types into time series data.

        Reads the original dataset, applies standardization, injects noise at each
        severity level, and saves the resulting corrupted datasets to CSV files.

        Output file types per level:
        - shift, spike, impulse, gaussian, missing, combined

        Parameters:
        - args: Argument object with fields:
            root_path, data_path, output_path (optional),
            spot_type, spot_n_points, spot_depth, spot_init_points, spot_init_level,
            zero_clip
        """
        df = pd.read_csv(os.path.join(args.root_path, args.data_path))
        output_path = getattr(args, 'output_path', None) or args.root_path
        os.makedirs(output_path, exist_ok=True)
        data_cols = df.columns[1:]
        X = df[data_cols].values.copy()
        self.evt_values = {}
        self.evt_for_col = [{} for _ in range(X.shape[1])]
        self.spot_args['type'] = args.spot_type
        self.spot_args['n_points'] = args.spot_n_points
        self.spot_args['depth'] = args.spot_depth
        self.spot_args['init_points'] = args.spot_init_points
        self.spot_args['init_level'] = args.spot_init_level

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if self.exp_spike_args_map.keys() != self.level_shift_args_map.keys():
            raise ValueError("The level shift and exponential spike arguments map keys must be the same.")

        for noise_level in self.exp_spike_args_map.keys():
            np.random.seed(self.seed)
            random.seed(self.seed)
            start_time = time.time()
            print(f'noise level: {noise_level}')

            file_name, file_ext = os.path.splitext(args.data_path)

            new_X_shift = np.zeros_like(X)
            new_X_spike = np.zeros_like(X)
            new_X_impulse = np.zeros_like(X)
            new_X_gaussian = np.zeros_like(X)
            new_X_missing = X.copy()
            new_X_combined = np.zeros_like(X)
            missing_masks = np.zeros(X.shape, dtype=bool)

            for i in range(X.shape[1]):
                print(f'column: {data_cols[i]}')
                self.evt_values = self.evt_for_col[i]
                X_col = X[:, i]

                results = self.inject_all_noise(X_col, noise_level)

                new_X_shift[:, i] = results['shift'] + X_col
                new_X_spike[:, i] = results['spike'] + X_col
                new_X_impulse[:, i] = results['impulse'] + X_col
                new_X_gaussian[:, i] = results['gaussian'] + X_col
                new_X_missing[:, i] = X_col
                missing_masks[:, i] = results['missing_mask']

                combined = results['shift'] + results['spike'] + results['impulse']
                new_X_combined[:, i] = combined + X_col

                if args.zero_clip:
                    new_X_shift[:, i] = np.clip(new_X_shift[:, i], 0, None)
                    new_X_spike[:, i] = np.clip(new_X_spike[:, i], 0, None)
                    new_X_impulse[:, i] = np.clip(new_X_impulse[:, i], 0, None)
                    new_X_gaussian[:, i] = np.clip(new_X_gaussian[:, i], 0, None)
                    new_X_combined[:, i] = np.clip(new_X_combined[:, i], 0, None)

                self.evt_for_col[i] = self.evt_values

            # Inverse transform all
            new_X_shift = scaler.inverse_transform(new_X_shift)
            new_X_spike = scaler.inverse_transform(new_X_spike)
            new_X_impulse = scaler.inverse_transform(new_X_impulse)
            new_X_gaussian = scaler.inverse_transform(new_X_gaussian)
            new_X_combined = scaler.inverse_transform(new_X_combined)
            new_X_missing = scaler.inverse_transform(new_X_missing)

            # Missing: set affected positions to zero in original scale
            new_X_missing[missing_masks] = 0
            if args.zero_clip:
                new_X_missing = np.clip(new_X_missing, 0, None)

            # Build DataFrames and save
            type_data_map = {
                'shift': new_X_shift,
                'spike': new_X_spike,
                'impulse': new_X_impulse,
                'gaussian': new_X_gaussian,
                'missing': new_X_missing,
                'combined': new_X_combined,
            }

            for noise_type, noise_data in type_data_map.items():
                out_name = f'{file_name}_level_{noise_level}_type_{noise_type}{file_ext}'
                out_df = df.copy()
                out_df[data_cols] = noise_data
                out_df.to_csv(os.path.join(output_path, out_name), index=False)

            end_time = time.time()
            print(f'noise level: {noise_level} done! time: {end_time - start_time:.2f}s')
        print('all done!')
