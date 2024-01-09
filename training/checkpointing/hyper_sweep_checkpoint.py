import json 

class HyperSweepCheckpoint:
    def __init__(self, hyper_set, completed_set):
        self.hyper_set = hyper_set
        self.completed_set = completed_set
    
    @classmethod
    def load(fpath):
        with json.load(fpath) as file:
            hp_checkpoint = HyperSweepCheckpoint(set(file["hyper_set"]), set(file["completed_set"]))
            return hp_checkpoint

    def save(self, fpath):
        hp_json = {
            "hyper_set": self.hyper_set,
            "completed_set": self.completed_set
        }

        json.dump(hp_json, fpath)