from training.utils import Config
from training.utils.collections import HyperParameterSet
from training.trainers.trainer import Trainer
from itertools import product
import torch.nn as nn
from torch.utils.data import Dataset

class HyperSweepManager:
    def __init__(
            self,
            model_class: type[nn.Module],
            optim_class: type[nn.Module],
            trainer_class: type[Trainer],
            dataset: Dataset,
            config: Config,
        ):
        self.__dict__.update(config.__dict__)
        self.dataset = dataset
        combs: list[tuple] = product(self.epochs, self.batch_sizes, self.shuffle, self.learning_rates)
        self.hyper_parameter_set: set[HyperParameterSet] = [HyperParameterSet(epoch_count=comb[0], learning_rate=comb[1], shuffle=comb[2], batch_size=comb[3]) for comb in combs]
        self.trainer_cls: type[Trainer] = trainer_class
        self.model_cls: type[nn.Module] = model_class
        self.optim_cls: type[nn.Module] = optim_class
    
    @classmethod
    def get_default_config():
        epochs = [50, 100, 250, 500, 1000]
        batch_sizes = [5, 10, 25, 50]
        learning_rates = [3e-3,1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 3e-1]
        shuffle_data = [True, False]

        return Config(
            epochs=epochs,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            shuffle_data=shuffle_data
        )

    def run_hyper_parameter_sweep(self):
        for params in self.hyper_parameter_set:
            model = self.model_cls()
            optimizer = self.optim_cls(params.learning_rate)

            trainer = Trainer(
                model,
                optimizer,
                params.epoch_count,
                self.dataset,
                params.batch_size,
                params.shuffle_data,
            )

            trainer.train()

            trainer.save_results()