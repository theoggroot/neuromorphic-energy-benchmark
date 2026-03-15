import torch
import torch.nn as nn

# Norse is a library for spiking neural networks
import norse.torch as norse


class SNN(nn.Module):

    """
    SNN = Spiking Neural Network

    Traditional neural networks use continuous values.

    Spiking networks use discrete spikes.

    Neurons only fire when membrane potential
    crosses a threshold.

    This mimics biological neurons.
    """

    def __init__(self):

        super().__init__()

        # input layer
        self.fc1 = nn.Linear(784, 256)

        # LIF neuron model
        #
        # LIF = Leaky Integrate and Fire
        #
        # neuron integrates incoming signals
        # when threshold reached → spike
        #
        self.lif = norse.LIFCell()

        # output layer
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        # flatten image
        x = x.view(-1, 784)

        # linear transform
        z = self.fc1(x)

        # spiking neuron dynamics
        z, _ = self.lif(z)

        # output layer
        z = self.fc2(z)

        return z