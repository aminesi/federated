import numpy as np

import config

NUM_CLIENTS = 100
TRAINING_FRACTION = .1
NUM_EPOCHS = 5
BATCH_SIZE = 10
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS = 100

if config.get_param('dataset') == 'adni':
    NUM_CLIENTS = 6
    TRAINING_FRACTION = 1
    NUM_EPOCHS = 1
    BATCH_SIZE = 32


def pick_clients(fraction: float):
    count = int(np.round(fraction * NUM_CLIENTS))
    return np.random.choice(range(NUM_CLIENTS), replace=False, size=count)
