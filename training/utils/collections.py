from collections import namedtuple

HyperParameters = namedtuple("HyperParameterSet", ["epoch_count", "learning_rate", "shuffle_data", "batch_size"])