import psutil
import os


def memory_usage():

    """
    Measures RAM usage of the running process.

    Why memory matters?

    Edge AI devices have limited memory.

    Neuromorphic systems aim to reduce memory usage.
    """

    # get current process
    process = psutil.Process(os.getpid())

    # memory in bytes
    mem = process.memory_info().rss

    # convert to MB
    mem = mem / (1024 * 1024)

    return mem