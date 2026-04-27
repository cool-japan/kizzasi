"""Type stubs for kizzasi._kizzasi (Rust extension module)."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

__version__: str

class Config:
    """Configuration for a Kizzasi signal predictor.

    Parameters
    ----------
    input_dim : int
        Dimension of input signal.
    output_dim : int
        Dimension of output prediction.
    hidden_dim : int
        Hidden state dimension (model capacity).
    num_layers : int
        Number of model layers.
    state_dim : int, optional
        SSM state dimension (default: 16).
    model_type : str, optional
        Architecture: "rwkv", "s4d", "transformer", "s4" (default: "s4d").
    """

    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    state_dim: int
    model_type: str

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        state_dim: int = 16,
        model_type: str = "s4d",
    ) -> None: ...

    @staticmethod
    def audio(sample_rate: int = 16000) -> Config:
        """Preset for audio signal prediction."""
        ...

    @staticmethod
    def robotics(state_dim: int = 6, action_dim: int = 6) -> Config:
        """Preset for robot control signals."""
        ...

    @staticmethod
    def sensor(num_sensors: int = 9) -> Config:
        """Preset for IoT sensor streams."""
        ...

    def __repr__(self) -> str: ...


class Predictor:
    """Autoregressive signal predictor.

    Parameters
    ----------
    config : Config
        Model configuration.
    """

    def __init__(self, config: Config) -> None: ...

    def step(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict one step ahead.

        Parameters
        ----------
        input : ndarray of shape (input_dim,), dtype float32
            Current signal sample.

        Returns
        -------
        ndarray of shape (output_dim,), dtype float32
            Next-step prediction.
        """
        ...

    def predict_n(
        self,
        input: NDArray[np.float32],
        n_steps: int,
    ) -> NDArray[np.float32]:
        """Predict n steps ahead autoregressively.

        Parameters
        ----------
        input : ndarray of shape (input_dim,), dtype float32
            Initial input.
        n_steps : int
            Number of prediction steps.

        Returns
        -------
        ndarray of shape (n_steps, output_dim), dtype float32
            Sequence of predictions.
        """
        ...

    def reset(self) -> None:
        """Reset internal hidden state to zero."""
        ...

    @property
    def input_dim(self) -> int: ...

    @property
    def output_dim(self) -> int: ...

    @property
    def hidden_dim(self) -> int: ...

    @property
    def model_type(self) -> str: ...

    def __repr__(self) -> str: ...
