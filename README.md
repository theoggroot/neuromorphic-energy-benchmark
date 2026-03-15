# Neuromorphic Energy Benchmark

A research-style benchmarking project comparing **Traditional Convolutional Neural Networks (CNNs)** and **Spiking Neural Networks (SNNs)** in terms of:

* Inference Latency
* Energy Consumption
* Memory Usage

This project demonstrates the motivation behind **neuromorphic computing** and energy-efficient AI.

---

# Project Motivation

Traditional deep learning models consume significant computational power.

Neuromorphic systems attempt to reduce energy consumption by mimicking biological neurons through **spike-based computation**.

This project benchmarks:

CNN vs SNN

to analyze their computational characteristics.

---

# Architecture

### CNN Architecture

Input (1×28×28)

→ Conv2D
→ ReLU
→ MaxPool

→ Conv2D
→ ReLU
→ MaxPool

→ Flatten

→ Fully Connected

→ Output (10 classes)

### SNN Architecture

Input (784)

→ Linear Layer

→ LIF Spiking Neuron

→ Linear Layer

→ Output

---

# Project Structure

```
neuromorphic-energy-benchmark
│
├── models
│   ├── cnn.py
│   └── snn.py
│
├── benchmark
│   ├── benchmark_runner.py
│   ├── latency.py
│   ├── energy.py
│   └── memory.py
│
├── analysis
│   ├── plot_results.py
│   └── architecture_diagram.py
│
├── utils
│   └── dataset.py
│
├── results
│   ├── logs
│   └── plots
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Benchmark Metrics

The benchmark measures:

| Metric  | Description                  |
| ------- | ---------------------------- |
| Latency | Time required for inference  |
| Energy  | Estimated energy consumption |
| Memory  | RAM usage during inference   |

Results are stored in:

```
results/logs/benchmark_results.csv
```

---

# Example Results

| Model | Latency  | Memory  |
| ----- | -------- | ------- |
| CNN   | ~0.75 ms | ~246 MB |
| SNN   | ~0.31 ms | ~247 MB |

---

# Generated Benchmark Plots

The project automatically generates plots:

* Latency comparison
* Memory consumption
* Energy estimation

Located in:

```
results/plots/
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/neuromorphic-energy-benchmark.git
```

Navigate into the project:

```
cd neuromorphic-energy-benchmark
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Run Benchmark

```
python main.py
```

This will:

1. Run CNN and SNN inference
2. Measure latency, energy, and memory
3. Save results
4. Generate plots

---

# Technologies Used

* Python
* PyTorch
* Norse (Spiking Neural Networks)
* Matplotlib
* Pandas

---

# Future Improvements

* CNN → SNN conversion benchmark
* Event-based neuromorphic datasets
* GPU benchmarking
* Energy per inference scaling

---

# License

MIT License

---

# Author

Developed as an exploration of **Neuromorphic Computing and Energy-Efficient AI Systems**.
