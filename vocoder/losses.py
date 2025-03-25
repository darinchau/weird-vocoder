import torch
from torch import nn
import torch.nn.functional as F
from .vggish import Vggish
from .stft import STFT


class AudioLoss(nn.Module):
    """A base class for all loss functions used in this project.
    The input should be two audio tensors under the same sample rate in either (C, T) or (B, C, T) format.
    The output should be a scalar tensor for (C, T) tensors or a tensor of shape (B,) for (B, C, T) tensors.

    If reduce is True, the output is always a scalar tensor. This is more common in training.

    Children classes implementing the forward method can assume only the BCT case and everything in the correct format"""

    def __init__(self, sample_rate: int, reduce: bool = True):
        super().__init__()
        self._sample_rate = sample_rate

    @property
    def sample_rate(self):
        return self._sample_rate

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, f"Expected tensors of the same shape, got {x.shape} and {y.shape}"
        has_batch = len(x.shape) == 3
        if not has_batch:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        assert x.shape[1] in (1, 2), f"Expected mono or stereo audio, got {x.shape[1]} channels"
        assert x.shape[2] == y.shape[2], f"Expected same number of frames, got {x.shape[2]} and {y.shape[2]}"
        result = super().__call__(x, y)

        assert len(result.shape) == 1, f"Expected a tensor of shape (B,), got {result.shape}"
        if not has_batch:
            result = result.squeeze(0)
        return result


class AudioMSELoss(AudioLoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))


class AudioWeightedNormLoss(AudioLoss):
    """Computes the loss

    F(x, y) = weight * ||x - y||_2 + (1 - weight) * ||x - y||_1
    """

    def __init__(self, sample_rate: int, reduce: bool = True, weight: float = 1.0):
        super().__init__(sample_rate, reduce)
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(x, y, reduction='none').mean(dim=(1, 2))
        l2 = F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))
        return self.weight * l2 + (1 - self.weight) * l1


class SpectrogramLoss(AudioLoss):
    def __init__(self, n: int, t: int, sample_rate: int, reduce: bool = True):
        super().__init__(sample_rate, reduce)
        self.stft = STFT(n, t)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.stft.forward(x)
        y = self.stft.forward(y)
        return F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))


class VggishLoss(AudioLoss):
    def __init__(self, sample_rate: int, reduce: bool = True):
        super().__init__(sample_rate, reduce)
        self.vggish = Vggish()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.vggish((x, self.sample_rate))
        y = self.vggish((y, self.sample_rate))
        return F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))


class DilatedVggishLoss(AudioLoss):
    def __init__(self, sample_rate: int, reduce: bool = True):
        super().__init__(sample_rate, reduce)
        self.vggish = Vggish()
        self.vggish.input_sr = self.sample_rate  # Don't resample to implicitly dilate the input

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.vggish((x, self.sample_rate))
        y = self.vggish((y, self.sample_rate))
        return F.mse_loss(x, y, reduction='none').mean(dim=(1, 2))
