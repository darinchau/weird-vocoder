
import torch


class STFT:
    """Try to enforce the constraint about the inputs and outputs of the stft such that input is always (B, L) and output is (B, N, T)"""

    def __init__(self, n: int, t: int):
        # Check that N must be one more than a power of 2 and n//4 > 1
        assert (n > 3) and (((n - 1) & (n - 2)) == 0), f"n must be one more than a power of 2, got {n}"
        self.n = n
        self.t = t
        self.n_fft = 2 * n - 2
        self.hop_length = n // 4

    @property
    def l(self):
        # Check that L is solvable: T = 1 + L // hop_length
        return (self.t - 1) * self.hop_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Expected input to have 2 dimensions, got {x.ndim}"
        assert x.shape[1] == self.l, f"Expected input to have shape (B, {self.l}), got {x.shape}"

        z = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=True,
            onesided=True,
        )

        assert isinstance(z, torch.Tensor), f"Expected output to be a tensor, got {type(z)}"
        assert z.shape == (x.shape[0], self.n, self.t), f"Expected output to have shape (B, {self.n // 2 + 1}, {self.t}), got {z.shape}"
        return z

    def inverse(self, z: torch.Tensor):
        assert z.ndim == 3, f"Expected input to have 3 dimensions, got {z.ndim}"
        assert z.shape[1] == self.n, f"Expected input to have shape (B, {self.n // 2 + 1}, {self.t}), got {z.shape}"
        assert z.shape[2] == self.t, f"Expected input to have shape (B, {self.n // 2 + 1}, {self.t}), got {z.shape}"
        assert torch.is_complex(z), f"Expected input to be complex, got {z.dtype}"

        # We expect no trimming or padding to be done, so length=None
        x = torch.istft(
            z,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(z.real),
            win_length=self.n_fft,
            normalized=True,
            center=True
        )

        assert x.shape == (z.shape[0], self.l), f"Expected output to have shape (B, {self.l}), got {x.shape}"
        return x
