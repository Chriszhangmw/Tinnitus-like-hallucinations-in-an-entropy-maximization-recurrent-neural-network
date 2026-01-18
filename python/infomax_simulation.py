"""Python translation of the MATLAB InfoMax auditory model.

This module consolidates the MATLAB scripts into a single Python implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import math
import os
import pickle
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NetworkConfig:
    eta_w: float = 0.1
    eta_k: float = 0.001


@dataclass
class SamplesConfig:
    step_samples: int = 1


@dataclass
class StepsConfig:
    ilearn: int = 1
    final_w_ilearn: int = 50000
    tinnitus_ilearn: int = 1050000
    nlearn: int = 2050000


@dataclass
class CostConfig:
    n_samp: int = 10
    n_calc: int = 100
    last: float = math.inf
    maxdiff: float = 100.0
    err: Optional[np.ndarray] = None


@dataclass
class TimeConfig:
    n_mean_time: int = 100
    dur: Optional[np.ndarray] = None


@dataclass
class DisplayConfig:
    disp: bool = False
    n_disp: int = 10
    write: bool = True
    n_write: int = 10


@dataclass
class FilesConfig:
    path: str = "Results"
    filenamebeg: str = "Auditory_"
    filename: str = ""
    duration_days: int = 1000
    n_save: int = 100


@dataclass
class AttenuationConfig:
    beta: float = 10.0
    f_0: int = 20
    minval: float = 0.0


@dataclass
class InputGenConfig:
    n_tones_max: int = 5
    a_min: float = 7.0
    a_max: float = 10.0
    noise_fac: float = 0.5
    white_noise: float = 0.5


@dataclass
class SamplesData:
    filename: str = "Samples"
    n_samples: int = 1000000
    x: Optional[np.ndarray] = None


@dataclass
class SimParams:
    net: "Infomax"
    Net: NetworkConfig
    Samples: SamplesConfig
    Steps: StepsConfig
    Cost: CostConfig
    Time: TimeConfig
    Display: DisplayConfig
    Files: FilesConfig
    Attenuate: AttenuationConfig
    quick: bool = False


class Infomax:
    """Implements an InfoMax network."""

    Default_FF_ridge = 0.0
    Default_Rec_ridge = 0.0
    Default_Beta = 0.0
    Default_niter = 3000
    Default_alpha = 0.01
    Default_tolfun = 1e-8

    def __init__(
        self,
        inputs: int,
        outputs: int,
        threshold: bool,
        FF_eta: float,
        Rec_eta: float,
        FF_ridge: Optional[float] = None,
        Rec_ridge: Optional[float] = None,
        beta: Optional[float] = None,
        niter: Optional[int] = None,
        alpha: Optional[float] = None,
        tolfun: Optional[float] = None,
    ) -> None:
        if inputs <= 0 or outputs <= 0:
            raise ValueError("inputs and outputs must be positive")
        if FF_eta < 0 or Rec_eta < 0:
            raise ValueError("Learning rate cannot be negative")

        self.Inputs = inputs
        self.Outputs = outputs
        self.Threshold = threshold
        self.FF_eta = FF_eta
        self.Rec_eta = Rec_eta
        self.FF_ridge = self.Default_FF_ridge if FF_ridge is None else FF_ridge
        self.Rec_ridge = self.Default_Rec_ridge if Rec_ridge is None else Rec_ridge
        self.Beta = self.Default_Beta if beta is None else beta
        self.niter = self.Default_niter if niter is None else niter
        self.alpha = self.Default_alpha if alpha is None else alpha
        self.tolfun = self.Default_tolfun if tolfun is None else tolfun

        self.W: np.ndarray
        self.K: np.ndarray
        self.ResetConnections()

    def Evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get output vector s for an input vector x."""
        x = _ensure_2d(x)
        x1 = x
        if self.Threshold:
            x1 = np.vstack([x, -np.ones((1, x.shape[1]))])

        g = np.zeros((self.Outputs, x1.shape[1]))
        gp = np.zeros_like(g)
        gpp = np.zeros_like(g)
        for i in range(x1.shape[1]):
            g[:, i], gp[:, i], gpp[:, i] = self._gcalc(x1[:, i])
        return g, gp, gpp

    def Learn(self, x: np.ndarray) -> None:
        """Learn from samples matrix x."""
        x = _ensure_2d(x)
        V = self.W
        R = self.K
        eta_W = self.FF_eta
        eta_K = self.Rec_eta
        beta = self.Beta

        if eta_W == 0 and eta_K == 0:
            return

        dW = np.zeros_like(V)
        dV = np.zeros_like(V)
        dK = np.zeros_like(R)

        noK = np.all(R == 0)

        N = self.Outputs
        M = self.Inputs
        I_N = np.eye(N)
        I_M = np.eye(M)

        for nsample in range(x.shape[1]):
            x1 = x[:, nsample]
            s, gp, gpp = self.Evaluate(x1)
            s = s[:, 0]
            gp = gp[:, 0]
            gpp = gpp[:, 0]

            Phi = np.diag(gp)
            if not noK:
                IGK = I_N - Phi @ R
                Phi = np.linalg.solve(IGK, Phi)

            Chi = Phi @ V[:, :M]
            Chipinv = np.linalg.pinv(Chi)
            Xi = I_N
            if N > M:
                Xi = Chi @ Chipinv
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma = np.diag(Xi @ Phi) * gpp / (gp**3)
                gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            if eta_W != 0:
                dV[:, :M] = Phi.T @ (Chipinv.T + np.outer(gamma, x1))
                if N < M and beta != 0:
                    x_perp = (I_M - Chipinv @ Chi) @ x1
                    dV[:, :M] += beta * Phi.T @ Chipinv.T @ np.outer(x1, x_perp)
                if self.Threshold:
                    dV[:, M] = -Phi.T @ gamma
                dW += dV
            if eta_K != 0:
                dK += Phi.T @ (Xi + np.outer(gamma, s))

        n_samples = x.shape[1]
        if eta_W != 0:
            if self.FF_ridge != 0:
                self.W = self.W - (eta_W * self.FF_ridge * np.sign(self.W))
            self.W = self.W + (eta_W / n_samples) * dW
        if eta_K != 0:
            if self.Rec_ridge != 0:
                self.K = self.K - (eta_K * self.Rec_ridge * self.K)
            np.fill_diagonal(dK, 0)
            self.K = self.K + (eta_K / n_samples) * dK

    def GetCost(self, x: np.ndarray) -> float:
        """Get the cost function for a given input."""
        x = _ensure_2d(x)
        cost = 0.0
        V = self.W
        R = self.K
        N = self.Outputs
        I = np.eye(N)
        noK = np.all(R == 0)

        for nsample in range(x.shape[1]):
            x0 = x[:, nsample]
            if self.Threshold:
                x0 = np.concatenate([x0, [-1.0]])
            _, gp, _ = self._gcalc(x0)
            Phi = np.diag(gp)
            if not noK:
                Phi = np.linalg.solve(I - Phi @ R, Phi)
            Chi = Phi @ V
            svd_vals = np.linalg.svd(Chi, compute_uv=False)
            cost += np.sum(np.log(svd_vals + 1e-12))

        return -cost / x.shape[1]

    def GetNIter(self, x: np.ndarray) -> int:
        """Iteratively solve the activation function."""
        x = _ensure_2d(x)
        x1 = x[:, 0]
        if self.Threshold:
            x1 = np.concatenate([x1, [-1.0]])

        beta = self.alpha
        tol = self.tolfun
        n = self.niter
        n_iter = n

        g = _g_f(np.zeros(self.Outputs))
        H = self.W @ x1
        R = self.K

        for i in range(n):
            h = H + R @ g
            g1 = _g_f(h)
            g = beta * g1 + (1 - beta) * g
            if np.max(np.abs(g1 - g)) < tol:
                n_iter = i + 1
                break

        return n_iter

    def GetPopulationVector(self, n_samples: int) -> float:
        """Get the population vector for no input and weak initial output."""
        sigma = 0.1
        beta = self.alpha
        tol = self.tolfun
        n = self.niter
        R = self.K
        pop_vec = np.zeros(n_samples)
        po = np.arange(1, self.Outputs + 1) / self.Outputs
        eipo = np.exp((2 * math.pi * 1j) * po)

        for s in range(n_samples):
            g = _g_f(np.zeros(self.Outputs)) + sigma * np.random.randn(self.Outputs)
            for _ in range(n):
                g1 = _g_f(R @ g)
                g = beta * g1 + (1 - beta) * g
                if np.max(np.abs(g1 - g)) < tol:
                    break
            pv = np.sum(g * eipo) / self.Outputs
            pop_vec[s] = abs(pv)

        return float(np.mean(pop_vec))

    def ResetConnections(self) -> None:
        """Reset all network's connections."""
        tmp = _tonotopic_map(self.Inputs, self.Outputs)
        self.W = tmp
        if self.Threshold:
            self.W = np.zeros((self.Outputs, self.Inputs + 1))
            self.W[:, : self.Inputs] = tmp
        self.K = np.zeros((self.Outputs, self.Outputs))

    @staticmethod
    def _gfunc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = _g_f(x)
        gp = g * (1 - g)
        gpp = gp * (1 - 2 * g)
        return g, gp, gpp

    def _gcalc(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = self.W @ x
        R = self.K
        if np.all(R == 0):
            return self._gfunc(H)

        tol = self.tolfun
        n = self.niter
        N = self.Outputs
        I = np.eye(N)

        g = _g_f(np.zeros(N))
        for _ in range(n):
            g1, gp1 = _g_f(H + R @ g, with_derivative=True)
            G = np.diag(gp1)
            psi = I - G @ R
            D = np.linalg.solve(psi, g1 - g)
            g = g + D
            if np.mean(np.abs(D)) < tol:
                break

        h = H + R @ g
        return self._gfunc(h)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    return x


def _normal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mean) / sigma) ** 2
    )


def _tonotopic_map(inputs: int, outputs: int) -> np.ndarray:
    f_num = 50
    f_min = 20
    f_max = f_min + (f_num - 1) * 10
    f_inp = np.linspace(f_min, f_max, inputs)
    a = 0.01

    W = np.ones((outputs, len(f_inp)))
    sigma = 1 * (f_inp[-1] - f_inp[0]) / (f_num - 1)
    f_out = np.linspace(f_inp[0], f_inp[-1], outputs)

    for i in range(outputs):
        W[i, :] = _normal_pdf(f_inp, f_out[i], sigma)

    return a * W / (np.sum(W) / outputs)


def _g_f(x: np.ndarray, with_derivative: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    g = 1.0 / (1.0 + np.exp(-x))
    if with_derivative:
        gp = g * (1 - g)
        return g, gp
    return g


def attenuate_inputs(inputs: np.ndarray, config: AttenuationConfig) -> np.ndarray:
    j = np.arange(1, inputs.shape[0] + 1)
    scale = config.minval + (1 - config.minval) * (
        1.0 / (1.0 + np.exp(-config.beta * (config.f_0 - j)))
    )
    return inputs * scale[:, None]


def generate_input(inputs: int, n_samples: int, config: InputGenConfig) -> np.ndarray:
    x = np.zeros((inputs, n_samples))
    f_inp = np.arange(1, inputs + 1)
    rng = np.random.default_rng()

    for sample_count in range(n_samples):
        n_tones = rng.integers(1, config.n_tones_max + 1)
        sound = inputs * rng.random(n_tones)
        sound_std = 0.5 * inputs * np.abs(rng.standard_normal(n_tones))
        amp = config.a_min + (config.a_max - config.a_min) * rng.random(n_tones)
        x1 = np.zeros(inputs)
        for k in range(n_tones):
            x1 += (
                amp[k]
                * math.sqrt(2 * math.pi)
                * sound_std[k]
                * _normal_pdf(f_inp, sound[k], sound_std[k])
            )
        x[:, sample_count] = x1

    noise = config.noise_fac * (2 * rng.random(x.shape) - 1)
    x = x + noise + config.white_noise
    return 0.5 * x / np.max(np.abs(x))


def load_samples(filename: str) -> tuple[SamplesData, bool]:
    fullname = f"{filename}.pkl"
    if os.path.exists(fullname):
        with open(fullname, "rb") as handle:
            samples = pickle.load(handle)
        if isinstance(samples, SamplesData):
            return samples, True
    return SamplesData(filename=filename), False


def save_samples(samples: SamplesData) -> None:
    with open(f"{samples.filename}.pkl", "wb") as handle:
        pickle.dump(samples, handle)


def get_samples(inputs: int, n_samples: Optional[int] = None) -> SamplesData:
    samples, loaded = load_samples("Samples")
    if loaded and samples.x is not None:
        if n_samples is None or samples.x.shape[1] == n_samples:
            return samples

    input_gen = InputGenConfig()
    samples = SamplesData(filename="Samples")
    if n_samples is not None:
        samples.n_samples = n_samples
    samples.x = generate_input(inputs, samples.n_samples, input_gen)
    samples.n_samples = samples.x.shape[1]
    save_samples(samples)
    return samples


def init_sim_params(
    inputs: int,
    outputs: int,
    results_path: str,
    quick: bool = False,
) -> SimParams:
    net_config = NetworkConfig()
    threshold = True
    ff_ridge = 0.001
    rec_ridge = 0.209
    beta = 0.0
    niter = 10000
    alpha = 0.2
    tolfun = 1e-8
    if quick:
        niter = 500
        alpha = 0.3
        tolfun = 1e-6

    infomax = Infomax(
        inputs,
        outputs,
        threshold,
        net_config.eta_w,
        net_config.eta_k,
        ff_ridge,
        rec_ridge,
        beta,
        niter,
        alpha,
        tolfun,
    )

    samples_config = SamplesConfig()
    steps = StepsConfig()
    if quick:
        steps.final_w_ilearn = 50
        steps.tinnitus_ilearn = 100
        steps.nlearn = 200
        samples_config.step_samples = 5
    cost = CostConfig(err=np.zeros(steps.nlearn))
    if quick:
        cost.n_calc = 20
    time_cfg = TimeConfig()
    time_cfg.dur = np.zeros(time_cfg.n_mean_time)
    display = DisplayConfig()
    files = FilesConfig(
        path=results_path,
        filenamebeg=f"Auditory_{inputs}x{outputs}_",
        filename=f"Auditory_{inputs}x{outputs}_",
    )
    attenuate = AttenuationConfig(f_0=inputs // 2)

    return SimParams(
        net=infomax,
        Net=net_config,
        Samples=samples_config,
        Steps=steps,
        Cost=cost,
        Time=time_cfg,
        Display=display,
        Files=files,
        Attenuate=attenuate,
        quick=quick,
    )


def load_results(files: FilesConfig, filename_end: str = "") -> Optional[SimParams]:
    os.makedirs(files.path, exist_ok=True)
    target_date = datetime.now().date()

    def _candidate(day: datetime.date) -> str:
        name = f"{files.filename}{day.strftime('%d-%b-%Y')}{filename_end}.pkl"
        return os.path.join(files.path, name)

    candidate = _candidate(target_date)
    if os.path.exists(candidate):
        return _load_sim_params(candidate)

    for delta in range(1, files.duration_days + 1):
        day = target_date - timedelta(days=delta)
        candidate = _candidate(day)
        if os.path.exists(candidate):
            return _load_sim_params(candidate)

    return None


def _load_sim_params(path: str) -> SimParams:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_sim_params(sim_params: SimParams, suffix: str = "") -> str:
    os.makedirs(sim_params.Files.path, exist_ok=True)
    filename = f"{sim_params.Files.filename}{datetime.now().strftime('%d-%b-%Y')}{suffix}.pkl"
    path = os.path.join(sim_params.Files.path, filename)
    with open(path, "wb") as handle:
        pickle.dump(sim_params, handle)
    return path


def get_cost(sim_params: SimParams, samples: SamplesData) -> float:
    return sim_params.net.GetCost(samples.x[:, : sim_params.Cost.n_samp])


def plot_cost_history(sim_params: SimParams, output_path: str) -> None:
    costs = sim_params.Cost.err[: sim_params.Steps.ilearn]
    steps = np.arange(1, len(costs) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, costs, linewidth=1)
    plt.xlabel("Learning step")
    plt.ylabel("Cost")
    plt.title("InfoMax cost history")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_auditory_model(
    sim_params: SimParams,
    samples: SamplesData,
    plot_path: Optional[str] = None,
) -> None:
    attenuated = False
    k_learn = False

    if sim_params.Steps.ilearn > sim_params.Steps.tinnitus_ilearn:
        x = attenuate_inputs(samples.x, sim_params.Attenuate)
        attenuated = True
        k_learn = True
    else:
        x = samples.x
        if sim_params.Steps.ilearn > sim_params.Steps.final_w_ilearn:
            sim_params.net.FF_eta = 0
            sim_params.net.Rec_eta = sim_params.Net.eta_k
            k_learn = True
        else:
            sim_params.net.FF_eta = sim_params.Net.eta_w
            sim_params.net.Rec_eta = 0

    while sim_params.Steps.ilearn <= sim_params.Steps.nlearn:
        start = time.time()

        if not attenuated and sim_params.Steps.ilearn > sim_params.Steps.tinnitus_ilearn:
            x = attenuate_inputs(samples.x, sim_params.Attenuate)
            attenuated = True
            save_sim_params(sim_params, suffix="_K_learned")
        elif not k_learn and sim_params.Steps.ilearn > sim_params.Steps.final_w_ilearn:
            sim_params.net.FF_eta = 0
            sim_params.net.Rec_eta = sim_params.Net.eta_k
            k_learn = True
            save_sim_params(sim_params, suffix="_W_learned")

        rng = np.random.default_rng()
        if sim_params.Samples.step_samples >= x.shape[1]:
            indices = np.arange(x.shape[1])
        else:
            indices = rng.choice(x.shape[1], sim_params.Samples.step_samples, replace=False)
        x0 = x[:, indices]

        sim_params.net.Learn(x0)

        if (sim_params.Steps.ilearn - 1) % sim_params.Cost.n_calc == 0:
            cost = get_cost(sim_params, samples)
            if (
                sim_params.Steps.ilearn > sim_params.Steps.tinnitus_ilearn
                and cost - sim_params.Cost.last > sim_params.Cost.maxdiff
            ):
                break
            sim_params.Cost.last = cost
        sim_params.Cost.err[sim_params.Steps.ilearn - 1] = sim_params.Cost.last

        sim_params.Time.dur[(sim_params.Steps.ilearn - 1) % sim_params.Time.n_mean_time] = (
            time.time() - start
        )

        if sim_params.Display.write and sim_params.Steps.ilearn % sim_params.Display.n_write == 0:
            mean_time = np.mean(sim_params.Time.dur[: min(sim_params.Steps.ilearn, sim_params.Time.n_mean_time)])
            time_left_s = int((sim_params.Steps.nlearn - sim_params.Steps.ilearn) * mean_time)
            time_left_m, time_left_s = divmod(time_left_s, 60)
            time_left_h, time_left_m = divmod(time_left_m, 60)
            time_left_d, time_left_h = divmod(time_left_h, 24)
            print(f"Recurrent ridge: {sim_params.net.Rec_ridge}")
            print(f"Ilearn = {sim_params.Steps.ilearn}:  E = {sim_params.Cost.last}")
            print(
                f"Time left (est.): {time_left_d}:{time_left_h:02d}:{time_left_m:02d}:{time_left_s:02d}\n"
            )

        sim_params.Steps.ilearn += 1

        if (sim_params.Steps.ilearn - 1) % sim_params.Files.n_save == 0:
            save_sim_params(sim_params)

    if plot_path:
        plot_cost_history(sim_params, plot_path)


def batch_sim(
    inputs: int = 40,
    outputs: int = 400,
    results_path: str = "Results",
    quick: bool = False,
    plot: bool = True,
) -> None:
    if quick:
        samples = get_samples(inputs, n_samples=500)
        ridge_values = np.array([0.2])
    else:
        samples = get_samples(inputs)
        ridge_values = np.concatenate(
            [
                np.arange(0.1, 0.176, 0.005),
                np.arange(0.176, 0.191, 0.001),
                np.arange(0.195, 0.221, 0.005),
                np.arange(0.221, 0.236, 0.001),
                np.arange(0.24, 0.301, 0.01),
            ]
        )

    for ridge in ridge_values:
        sim_params = init_sim_params(inputs, outputs, results_path, quick=quick)
        sim_params.Files.filename = f"{sim_params.Files.filenamebeg}RidgeK_{ridge}_"
        sim_params.net.Rec_ridge = ridge

        loaded = load_results(sim_params.Files)
        if loaded:
            sim_params = loaded

        save_sim_params(sim_params)
        plot_path = None
        if plot:
            plot_path = os.path.join(
                results_path,
                f"cost_RidgeK_{ridge}.png",
            )
        run_auditory_model(sim_params, samples, plot_path=plot_path)


if __name__ == "__main__":
    batch_sim(
        inputs=40,
        outputs=400,
        results_path="Results",
        quick=True,
        plot=True,
    )
