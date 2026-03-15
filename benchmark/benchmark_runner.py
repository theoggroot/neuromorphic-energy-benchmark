import torch
import csv
import os

from models.cnn import CNN
from models.snn import SNN

from benchmark.latency import measure_latency
from benchmark.memory import memory_usage
from benchmark.energy import measure_energy


def run_benchmark():

    """
    This function performs the full benchmark
    and saves results to CSV.

    CSV results allow later analysis and plotting.
    """

    # Create models
    cnn = CNN()
    snn = SNN()

    # Dummy input image
    input_data = torch.randn(1, 1, 28, 28)

    # Create results directory if not present
    os.makedirs("results/logs", exist_ok=True)

    results = []

    # -------------------------------------------------
    # CNN Benchmark
    # -------------------------------------------------

    cnn_latency = measure_latency(cnn, input_data)

    cnn_energy = measure_energy(lambda: cnn(input_data))

    cnn_memory = memory_usage()

    results.append([
        "CNN",
        cnn_latency,
        cnn_energy,
        cnn_memory
    ])

    # -------------------------------------------------
    # SNN Benchmark
    # -------------------------------------------------

    snn_latency = measure_latency(snn, input_data)

    snn_energy = measure_energy(lambda: snn(input_data))

    snn_memory = memory_usage()

    results.append([
        "SNN",
        snn_latency,
        snn_energy,
        snn_memory
    ])

    # -------------------------------------------------
    # Save results to CSV
    # -------------------------------------------------

    file_path = "results/logs/benchmark_results.csv"

    with open(file_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Latency(ms)",
            "Energy(uJ)",
            "Memory(MB)"
        ])

        writer.writerows(results)

    print("Benchmark completed.")
    print("Results saved to:", file_path)