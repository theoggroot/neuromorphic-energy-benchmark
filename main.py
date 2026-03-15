"""
MAIN ENTRY POINT OF THE PROJECT

This script controls the entire benchmarking pipeline.

What this script does:

1. Runs the CNN vs SNN benchmark
2. Measures performance metrics:
      - Latency
      - Energy usage
      - Memory usage
3. Saves results into CSV file
4. Generates plots (graphs) from those results

Why is this important?

In research projects, reproducibility is critical.
Anyone should be able to clone the repository and
run ONE command to reproduce the benchmark results.

That is exactly what this script provides.
"""

# -------------------------------------------------------
# Import benchmark runner
# -------------------------------------------------------

# This module runs the actual experiments
# comparing CNN and SNN models
from benchmark.benchmark_runner import run_benchmark


# -------------------------------------------------------
# Import subprocess
# -------------------------------------------------------

# subprocess allows Python to run another script
# from inside this script.
#
# We use it to automatically run the plotting script
# after the benchmark finishes.
import subprocess


# -------------------------------------------------------
# Main function
# -------------------------------------------------------

def main():

    """
    Main control function of the project.

    This function executes the whole pipeline step-by-step.
    """

    print("\n=====================================")
    print(" Neuromorphic Energy Benchmark ")
    print("=====================================\n")

    # ---------------------------------------------------
    # STEP 1: Run Benchmark
    # ---------------------------------------------------
    #
    # This runs CNN and SNN inference
    # and measures:
    #
    #   • latency
    #   • energy consumption
    #   • memory usage
    #
    # The results are saved into:
    #
    # results/logs/benchmark_results.csv
    #
    print("Running CNN vs SNN benchmark...\n")

    run_benchmark()

    print("\nBenchmark finished successfully!\n")


    # ---------------------------------------------------
    # STEP 2: Generate graphs
    # ---------------------------------------------------
    #
    # After collecting benchmark results,
    # we want to visualize them.
    #
    # The plotting script reads the CSV file
    # and generates research-style graphs:
    #
    #   energy comparison
    #   latency comparison
    #   memory usage comparison
    #
    print("Generating result graphs...\n")

    subprocess.run(["python", "analysis/plot_results.py"])


    # ---------------------------------------------------
    # STEP 3: Notify user
    # ---------------------------------------------------

    print("\n=====================================")
    print(" Benchmark Completed Successfully ")
    print("=====================================\n")

    print("Results saved in:")
    print("   results/logs/benchmark_results.csv\n")

    print("Graphs generated in:")
    print("   results/plots/\n")

    print("You can now use these plots in:")
    print("   • GitHub README")
    print("   • research reports")
    print("   • presentations\n")


# -------------------------------------------------------
# Python entry point
# -------------------------------------------------------
#
# This condition ensures the script runs only when
# executed directly, not when imported as a module.
#
# Example:
#
# python main.py
#
if __name__ == "__main__":
    main()