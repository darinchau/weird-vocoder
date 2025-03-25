from AutoMasher.fyp import Audio, SongDataset
from .constants import *
import torch
import gc


def get_test_audio() -> Audio:
    return Audio.load(TEST_AUDIO_PATH)


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
