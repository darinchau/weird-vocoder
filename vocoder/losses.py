import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce
from .vggish import Vggish
from .stft import make_stft, STFT


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class AudioLoss(nn.Module):
    """A base class for all loss functions used in this project.
    The input should be two audio tensors under the same sample rate in either (T) or (B, T) format.
    The output should be a scalar tensor for (T, ) tensors or a tensor of shape (B,) for (B, T) tensors.

    If reduce is True, the output is always a scalar tensor (with mean reduction). This is more common in training.

    Children classes implementing the forward method can assume only the BT case and everything in the correct format"""

    def __init__(self, audio_length: int, sample_rate: int, reduce: bool = True):
        super().__init__()
        self._sample_rate = sample_rate
        self._audio_length = audio_length
        self.__super_inited = True
        self.reduce = reduce

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def audio_length(self):
        return self._audio_length

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.__super_inited:
            raise RuntimeError("AudioLoss not properly initialized")

        assert x.shape == y.shape, f"Expected tensors of the same shape, got {x.shape} and {y.shape}"

        has_batch = len(x.shape) == 2
        if not has_batch:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        result = super().__call__(x, y)

        assert len(result.shape) == 1, f"Expected a tensor of shape (B,), got {result.shape}"
        if self.reduce:
            result = result.mean()
        elif not has_batch:
            result = result.squeeze(0)
        return result


class AudioMSELoss(AudioLoss):
    def __init__(self, audio_length: int, sample_rate: int, reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y, reduction='none').mean(dim=1)


class AudioWeightedNormLoss(AudioLoss):
    """Computes the loss

    F(x, y) = weight * ||x - y||_2 + (1 - weight) * ||x - y||_1
    """

    def __init__(self, audio_length: int, sample_rate: int, reduce: bool = True, weight: float = 1.0):
        super().__init__(audio_length, sample_rate, reduce)
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(x, y, reduction='none').mean(dim=1)
        l2 = F.mse_loss(x, y, reduction='none').mean(dim=1)
        return self.weight * l2 + (1 - self.weight) * l1


class VggishLoss(AudioLoss):
    def __init__(self, audio_length: int, sample_rate: int, reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)
        self.vggish = Vggish()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.vggish((x, self.sample_rate))
        y = self.vggish((y, self.sample_rate))
        return F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))


class DilatedVggishLoss(AudioLoss):
    def __init__(self, audio_length: int, sample_rate: int, reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)
        self.vggish = Vggish()
        self.vggish.input_sr = self.sample_rate  # Don't resample to implicitly dilate the input

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.vggish((x, self.sample_rate))
        y = self.vggish((y, self.sample_rate))
        return F.mse_loss(x, y, reduction='none').mean(dim=(1))


### STFT losses ###

class STFTLoss(AudioLoss):
    def __init__(self, audio_length: int, sample_rate: int, nfft: int, hop_length: int | None = None, t: int | None = None, reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)
        self.stft = make_stft(audio_length=audio_length, n=nfft, hop_length=hop_length, t=t)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(
            torch.view_as_real(self.stft.forward(x)),
            torch.view_as_real(self.stft.forward(y)),
            reduction='none'
        ).mean(dim=(1, 2, 3))


class MultiResolutionSTFTLoss(AudioLoss):
    def __init__(self, audio_length: int, sample_rate: int, windows: list[int], weights: list[float], reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)
        assert all(w > 0 for w in windows), "All window sizes must be positive"
        assert all(w > 0 for w in weights), "All weights must be positive"
        assert len(windows) == len(weights), "The number of windows and weights must be the same"
        assert all(_is_power_of_2(w) for w in windows), "All window sizes must be a power of 2"
        assert all(audio_length % w == 0 for w in windows), "All window sizes must divide the audio length"
        self.stfts = [
            STFT(f * 2 + 1, 2 * audio_length // f + 1)
            for f in windows
        ]
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return reduce(lambda a, b: a + b, [
            F.mse_loss(
                torch.view_as_real(stft.forward(x)),
                torch.view_as_real(stft.forward(y)),
                reduction='none').mean(dim=(1, 2, 3)) * w
            for stft, w in zip(self.stfts, self.weights)  # type: ignore
        ])


class MultiDilationSTFTLoss(AudioLoss):
    """STFT loss with a single window but different dilations.

    Example: STFT(131072, 44100, 128, [1, 2, 4])"""

    def __init__(self, audio_length: int, sample_rate: int, window: int, weights: list[float], reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)
        assert window > 0, "The window size must be positive"
        assert _is_power_of_2(window), "The window size must be a power of 2"
        assert audio_length % window == 0, "The window size must divide the audio length"
        assert len(weights) == 3
        self.stfts = [
            STFT(1 * window + 1, 4 * audio_length // window + 1),
            STFT(2 * window + 1, 2 * audio_length // window + 1),
            STFT(4 * window + 1, 1 * audio_length // window + 1),
        ]
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        for stft, w in zip(self.stfts, self.weights):  # type: ignore
            x_ = torch.view_as_real(stft.forward(x))
            y_ = torch.view_as_real(stft.forward(y))
            loss += F.mse_loss(x_, y_, reduction='none').mean(dim=(1, 2, 3)) * w
        return loss


class MelReconstructionLoss(AudioLoss):
    def __init__(self, audio_length: int, sample_rate: int, nfft: int, n_mels: int, reduce: bool = True):
        super().__init__(audio_length, sample_rate, reduce)
        self.stft = make_stft(audio_length=audio_length, n=nfft)
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.stft.mel(x, self.sample_rate, nmels=self.n_mels)
        y = self.stft.mel(y, self.sample_rate, nmels=self.n_mels)
        return F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))


class LowHopSTFTLoss(STFTLoss):
    def __init__(self, audio_length: int, sample_rate: int, nfft: int, hop_length: int = 8, reduce: bool = True):
        super().__init__(audio_length, sample_rate, nfft, hop_length=hop_length, reduce=reduce)
