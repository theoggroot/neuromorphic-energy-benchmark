import pandas as pd
import matplotlib.pyplot as plt
import os

# Create plot directory
os.makedirs("results/plots", exist_ok=True)

# Load benchmark results
data = pd.read_csv("results/logs/benchmark_results.csv")

models = data["Model"]
latency = data["Latency(ms)"]
energy = data["Energy(uJ)"]
memory = data["Memory(MB)"]

# ------------------------------------------------
# Energy Plot
# ------------------------------------------------

plt.figure()

plt.bar(models, energy)

plt.ylabel("Energy (microjoules)")

plt.title("Energy Consumption Comparison")

plt.savefig("results/plots/energy_plot.png")

plt.close()

# ------------------------------------------------
# Latency Plot
# ------------------------------------------------

plt.figure()

plt.bar(models, latency)

plt.ylabel("Latency (ms)")

plt.title("Inference Latency Comparison")

plt.savefig("results/plots/latency_plot.png")

plt.close()

# ------------------------------------------------
# Memory Plot
# ------------------------------------------------

plt.figure()

plt.bar(models, memory)

plt.ylabel("Memory Usage (MB)")

plt.title("Memory Consumption Comparison")

plt.savefig("results/plots/memory_plot.png")

plt.close()

print("Plots saved in results/plots/")