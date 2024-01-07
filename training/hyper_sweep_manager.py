from training.utils import Config
from training.utils.collections import HyperParameterSet
from itertools import product


class HyperSweepManager:
    def __init__(self, config):
        self.__dict__.update(config)

        combs = product(self.learning_rates, self.epochs, self.batch_sizes, self.shuffle)
        [HyperParameterSet(epoch_count=comb[0], learning_rate=comb[]) for comb in combs]
    
    @classmethod
    def get_default_config():
        epochs = [50, 100, 250, 500, 1000]
        batch_sizes = [5, 10, 25, 50]
        learning_rates = [3e-3,1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 3e-1]
        shuffle = [True, False]

        return Config(
            epochs=epochs,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            shuffles=shuffle
        )

    def run_hypersweeps():