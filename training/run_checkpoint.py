import torch
import os

class RunCheckpoint:
    def __init__(
            self,
            model,
            learning_rate,
            optimizer,
            iter_num,
            epoch_count,
            batch_size
        ):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.iter_num = iter_num
        self.epoch_count = epoch_count
        self.batch_size = batch_size

    @classmethod
    def load(file_path, device):
        save_data = torch.load(file_path, device)
        
        model = save_data["model"]
        learning_rate = save_data["learning_rate"]
        optimizer = save_data["optim"]
        iter_num = save_data["iter_num"]
        epoch_count = save_data["epoch_count"]
        batch_size = save_data["batch_size"]

        return RunCheckpoint(
            model,
            learning_rate,
            optimizer,
            iter_num,
            epoch_count,
            batch_size
        )
    
    def save(self, fpath):
        model_info = {
            "model": self.model.save_dict(),
            "optim": self.optimizer.save_dict(),
            "learning_rate": self.learning_rate, 
            "iter_num": self.iter_num,
            "epoch_count": self.epoch_count,
            "batch_size": self.batch_size
        }
        folder = f"{self.iter_num}_{self.epoch_count}_{self.batch_size}_{str(self.learning_rate).strip('.')}"

        fpath = os.join(fpath, folder, "checkpoint.pt")
        torch.save(model_info, fpath)