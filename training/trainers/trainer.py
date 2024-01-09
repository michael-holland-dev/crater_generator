from training import TrainCheckpoint
from training.utils import Config
import torch
import os
import torch.nn as nn

class Trainer:
    def __init__(
            self,
            model,
            optim,
            dataset,
            device,
            training_config: Config
        ):
        self.model: nn.Module = model
        self.optimizer = optim
        self.dataset = dataset
        self.device = device
        
        self.__update_config(training_config)

    def __update_config(self, config: Config):
        
        self.__dict__.update(config.__dict__)

    def train(self):
        raise NotImplementedError("Training Function not Implemented")
    
    def _save_checkpoint(self, model, optimizer, epoch, iter_num, batch_size, shuffle):
        # Implement Save Checkpointing
        checkpoint = TrainCheckpoint(self.model, self.optimizer, )
        checkpoint.save()
    
    def _load_checkpoint(self):
        # Implement Load Checkpointing
        pass

    def save_results(self, path: str):
        # Implement Final Training Results Checkpoint
        if not os.path.exists(path):
            os.mkdir(path)

        model_stats = {
            "model_weights": self.model.state_dict()
        }

        torch.save(model_stats, os.join(path, "path.pt"))