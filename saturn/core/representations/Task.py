import torch
import torch.nn as nn
import torch.utils.data as data
from collections import namedtuple
import typing
from typing import List, Dict
from .Strategy import Strategy
import string
import random
import os


class HParams:
    """Training specs for a task.
    
    Parameters:
        lr: Learning Rate.
        
        epochs: epoch count (i.e. how many passes of the dataloader to complete). Should not be set if batch_count is configured.
        
        batch_count: Optional parameter that can be passed instead of epochs. Useful for partial epochs (e.g. by profiler).
        Should not be set if epochs is configured.
        
        optimizer_cls: Optimizer class to instantiate and use for training.
        
        kwargs: Any additional parameters to pass to your get_model function during instantiation.
        
        
    """

    def __init__(self, lr: float, epochs: int = None, batch_count: int = None, optimizer_cls: torch.optim.Optimizer = None, **kwargs) -> None:
        if (batch_count is not None and epochs is not None) or (batch_count is None and epochs is None):
            raise ValueError("""Epoch count and batch count cannot both be entered at the same time.
                            Only one or the other should be set.""")
        self.lr = lr
        self.epochs = epochs
        self.batch_count = batch_count
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

    def as_dict(self):
        return {"lr": self.lr, "epochs": self.epochs, "batch_count": self.batch_count}

    def __str__(self) -> str:
        if self.epochs is not None:
            return ("----Task Hyperparameters----\n"
                    "\t\tLearning Rate: {}\n"
                    "\t\tEpochs: {}".format(self.lr, self.epochs))
        else:
            return ("----Task Hyperparameters----\n"
                    "\t\tLearning Rate: {}\n"
                    "\t\tBatch Count: {}".format(self.lr, self.batch_count))


class Task:
    """Model task.

    Consists of a model, a dataloader, and various training hparams.
    Optional name parameter can help with debugging and ID'ing models.
    
    Parameters:
        get_model: Model loading/initialization function to create the initial model.
        
        get_dataloader: Data-loader construction function.
        
        loss_function: The loss function to be used during training.
        
        hparams: Instance of the HParams class, used to specify any hyperparameters for training.
        
        gpu_range: By default, Saturn will consider assigning any number of GPUs to your job (e.g. 1-8 on an 8-GPU node).
        However, to limit the search space and accelerate solving, you may wish to manually limit the search to certain configurations.
        Setting the GPU range variable to a list of valid selections will allow this.
        
        name: Used to identify the model and determine its filename when saving.
        
        hints: A dictionary of key-value pairs that will be passed to your executor code to guide execution.
        
        save_dir: Where to save model files.
    
    """

    """
        DO NOT pre instantiate models in memory. Will lead
        to DRAM capacity issues, ideally we should be able to instantiate
        as we need to.
    
    """

    def __init__(self, get_model, get_dataloader, loss_function: typing.Callable, hparams: HParams,
                                gpu_range: List[int] = None, name: str = None, hints=None, save_dir="./saved_models"):
        self.hints = hints
        self.internal_get_model = get_model
        self.internal_dl = get_dataloader
        self.hparams = hparams
        self.loss_function = loss_function
        self.gpu_range = gpu_range
        if name is None:
            name = ''.join(random.choices(string.ascii_uppercase +
                                          string.digits, k=32))

        self.saved_dataloader = None
        self.name = name
        self.save_dir = save_dir
    
        self.strategies = {}
        self.selected_strategy = None

        if self.hints is not None:
            if self.hints.get("is_transformer", False) and (self.hints.get("transformer_cls", None) is None or not isinstance(self.hints["transformer_cls"], set)):
                raise ValueError(
                    """If a model is flagged as a Transformer, you must specify your 'attention block' class in a dict passed as the transformer_cls parameter to Task.""")

       
        self.epoch_length = len(self.internal_dl())
        self.total_batches = self.epoch_length * self.hparams.epochs if self.hparams.epochs else self.hparams.batch_count

        self.current_batch = 0

    def get_iterator(self, modified_dl=None):
        if modified_dl is not None:
            dl = iter(modified_dl)
         
        else:
            dl = iter(self.internal_dl())
        for _ in range(self.current_batch):
            next(dl)
        return dl

    def get_fresh_iterator(self):
        return iter(self.internal_dl())
    
    def change_name(self, name=None):
        if name is None:
            self.name = ''.join(random.choices(string.ascii_uppercase +
                                              string.digits, k=32))

    def save(self, model):
        print("Saving model {}".format(self.name))
        torch.save(model, ".models/{}.pt".format(self.name))
        print("Saved model {}".format(self.name))

    def reconfigure(self, batch_count):
        self.current_batch = (self.current_batch +
                              batch_count) % self.epoch_length

    def has_ckpt(self):
        return os.path.isfile('.models/{}.pt'.format(self.name))

    def get_model(self, fresh=False):
        if os.path.isfile('.models/{}.pt'.format(self.name)) and not fresh:
            return torch.load(".models/{}.pt".format(self.name))
        else:
            return self.internal_get_model(self.hparams.kwargs)

    def select_strategy(self, strat: Strategy):
        self.selected_strategy = strat

    def __str__(self) -> str:
        return ("----Task {}----\n"
                "\t{}\n"
                "\tCandidate Strategies: {}\n"
                "\tSelected Strategy: {}\n"
                .format(self.name, self.hparams, self.strategies, self.selected_strategy))
