from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):

    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, y):
        """Encode 'y' into latent features 'z'."""
        raise NotImplementedError

    def decode(self, z):
        """Decode latent features 'z' back to 'y'."""
        raise NotImplementedError

    def predict(self, x, **kwargs):
        """Given condition 'x', use decoder to predict 'y'."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass