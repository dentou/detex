import torch
from torch import nn as nn


class BaseWrapper(nn.Module):
    def __init__(self, model, **kkwargs):
        super().__init__()
        self.model = model
    
    def make_blackbox(self, **kwargs) -> "Callable":
        """Implement this in subclass
        Return a forward function
        """
        raise NotImplementedError