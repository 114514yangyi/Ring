from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MadgwickConfig:
    beta: float = 0.041
    dt: float = 0.005
    gravity: float = 9.81


def cross_correlation_offset(
    clean_signal: np.ndarray, raw_signal: np.ndarray
) -> int:
    """Return the raw index where the clean signal starts (may be negative)."""

    clean = np.asarray(clean_signal, dtype=np.float64)
    raw = np.asarray(raw_signal, dtype=np.float64)
    clean = clean - clean.mean()
    raw = raw - raw.mean()
    size = len(clean) + len(raw)
    correlation = np.fft.irfft(
        np.fft.rfft(clean, size) * np.conj(np.fft.rfft(raw, size)), size
    )
    lag = int(np.argmax(correlation))
    if lag > len(raw):
        lag -= size
    return -lag


def _gravity_direction(quaternion: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quaternion
    return np.array(
        [
            2 * (q1 * q3 - q0 * q2),
            2 * (q0 * q1 + q2 * q3),
            1 - 2 * (q1 * q1 + q2 * q2),
        ]
    )


def madgwick_linear_acc(
    acc: np.ndarray, gyro: np.ndarray, config: MadgwickConfig | None = None
) -> np.ndarray:
    """Gravity-removed acceleration in the sensor frame (Madgwick AHRS)."""

    config = config or MadgwickConfig()
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    if acc.shape != gyro.shape or acc.shape[1] != 3:
        raise ValueError("acc and gyro must both have shape [frames, 3]")

    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    linear = np.empty_like(acc)
    for index in range(len(acc)):
        q0, q1, q2, q3 = quaternion
        gx, gy, gz = gyro[index]
        q_dot = 0.5 * np.array(
            [
                -q1 * gx - q2 * gy - q3 * gz,
                q0 * gx + q2 * gz - q3 * gy,
                q0 * gy - q1 * gz + q3 * gx,
                q0 * gz + q1 * gy - q2 * gx,
            ]
        )

        norm = np.linalg.norm(acc[index])
        if norm > 1e-6:
            ax, ay, az = acc[index] / norm
            q0, q1, q2, q3 = quaternion
            objective = np.array(
                [
                    2 * (q1 * q3 - q0 * q2) - ax,
                    2 * (q0 * q1 + q2 * q3) - ay,
                    2 * (0.5 - q1 * q1 - q2 * q2) - az,
                ]
            )
            jacobian = np.array(
                [
                    [-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                    [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                    [0.0, -4 * q1, -4 * q2, 0.0],
                ]
            )
            gradient = jacobian.T @ objective
            gradient = gradient / (np.linalg.norm(gradient) + 1e-12)
            q_dot = q_dot - config.beta * gradient

        quaternion = quaternion + q_dot * config.dt
        quaternion = quaternion / np.linalg.norm(quaternion)

        linear[index] = (
            acc[index] - config.gravity * _gravity_direction(quaternion)
        )
    return linear
