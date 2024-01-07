import itertools
import torch
import os

class TrainingManager:
    def __init__(self, model, optimizer, checkpointing=True):
        self.model = model
        self.optimizer = optimizer
    
    def 

    def train(self, epoch_count, batch_size, learning_rate):

    def hyperparameter_sweep(self, epochs, batch_sizes, learning_rates):
        itertools.product(epochs, batch_sizes, learning_rates)