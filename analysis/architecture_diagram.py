import sys
import os
import torch
from torchviz import make_dot

# Fix import path so models can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cnn import CNN
from models.snn import SNN


def generate_architecture():

    # Ensure output folder exists
    os.makedirs("results/plots", exist_ok=True)

    # Dummy input tensor
    x = torch.randn(1, 1, 28, 28)

    # ---------- CNN ----------
    cnn = CNN()
    cnn_out = cnn(x)

    cnn_graph = make_dot(cnn_out, params=dict(cnn.named_parameters()))
    cnn_graph.format = "png"

    cnn_graph.render("results/plots/cnn_architecture", cleanup=True)

    print("CNN architecture saved")

    # ---------- SNN ----------
    snn = SNN()
    snn_out = snn(x)

    snn_graph = make_dot(snn_out, params=dict(snn.named_parameters()))
    snn_graph.format = "png"

    snn_graph.render("results/plots/snn_architecture", cleanup=True)

    print("SNN architecture saved")


if __name__ == "__main__":
    generate_architecture()