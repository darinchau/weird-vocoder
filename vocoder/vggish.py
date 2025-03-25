# Implements VGGish feature extractor for Audio
import torch
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
from AutoMasher.fyp import Audio
import torchaudio.functional as F


class Vggish:
    """Wraps a VGGish model for feature extraction.

    Input: Audio or tuple[torch.Tensor, int]: Audio or a tuple of waveform and sample rate. The waveform must be in the range [-1, 1] and the tensor must have shape (C, T) or (B, C, T).
    Output: torch.Tensor: The extracted features. The tensor has shape (B, C, D) or (C, D) where B is the batch size, C is the number of channels, and D is the number of frames.
    """

    def __init__(self):
        self.input_sr = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.model = VGGISH.get_model()

    def __call__(self, audio: Audio | tuple[torch.Tensor, int]):
        if isinstance(audio, Audio):
            x = audio.resample(self.input_sr).data
        else:
            y, sr = audio
            x = F.resample(y, sr, self.input_sr)
        assert len(x.shape) in (2, 3), f"Expected 2D or 3D tensor, got {x.shape}"
        has_batch = len(x.shape) == 3
        if has_batch:
            b, c, t = x.shape
            x = x.view(-1, t)
        else:
            # To apease the typechecker
            b = 1
            c, t = x.shape
        feats = torch.stack([self.model(self.input_proc(w)) for w in x]).flatten(-2, -1)
        if has_batch:
            feats = feats.view(b, c, -1)
        return feats

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self
