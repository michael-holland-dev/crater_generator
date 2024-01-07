from collections import namedtuple

HyperParameters = namedtuple("HyperParameterSet", ["epoch_count", "learning_rate", "batch_size", "shuffle"])