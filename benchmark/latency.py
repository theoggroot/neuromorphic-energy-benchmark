import time

def measure_latency(model, input_data, runs=100):

    """
    Measures average inference latency.

    We run the model multiple times because
    a single run is too fast to measure accurately.
    """

    start = time.time()

    for _ in range(runs):
        model(input_data)

    end = time.time()

    total_time = end - start

    avg_latency = (total_time / runs) * 1000

    return avg_latency