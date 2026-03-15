"""
Energy measurement module.

pyRAPL only works on Linux systems with Intel RAPL support.

If the system does not support pyRAPL (like Windows),
the script will fall back to a simple timing-based estimate.
"""

import time
import platform

try:
    import pyRAPL
    pyRAPL.setup()
    RAPL_AVAILABLE = True
except:
    RAPL_AVAILABLE = False


def measure_energy(function):

    """
    Measures energy consumption of a function.

    If pyRAPL is available (Linux):
        returns real CPU energy measurement

    If not (Windows):
        returns estimated energy based on runtime
    """

    # ------------------------------------------------
    # Case 1: Real energy measurement (Linux)
    # ------------------------------------------------
    if RAPL_AVAILABLE and platform.system() == "Linux":

        meter = pyRAPL.Measurement('energy')

        meter.begin()

        function()

        meter.end()

        return meter.result.pkg

    # ------------------------------------------------
    # Case 2: Fallback estimate (Windows)
    # ------------------------------------------------
    else:

        start = time.time()

        function()

        end = time.time()

        runtime = end - start

        # simple estimate: assume 65W CPU
        estimated_energy = runtime * 65 * 1000000

        return estimated_energy