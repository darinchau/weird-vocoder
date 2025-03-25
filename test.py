from vocoder.stft import STFT
from vocoder.util import get_test_audio


def test_stft():
    values_to_test = [
        (513, 128),
        (257, 256),
        (129, 512),
        (65, 1024),
        (33, 2048),
        (17, 4096),
        (9, 8192)
    ]
    for n, t in values_to_test:
        audio = get_test_audio()
        stft = STFT(n, t)
        data = audio.slice_frames(0, stft.l).data
        spec = stft.forward(data)
        aud = stft.inverse(spec)
        assert (aud - data).abs().max() < 1e-6


if __name__ == "__main__":
    test_stft()
