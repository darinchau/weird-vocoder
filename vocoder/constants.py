# This specifies the constants used in the project and the structure of the dataset
import os
from math import isclose

BEAT_MODEL_PATH = "./AutoMasher/resources/ckpts/beat_transformer.pt"
CHORD_MODEL_PATH = "./AutoMasher/resources/ckpts/btc_model_large_voca.pt"

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)
