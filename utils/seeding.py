"""
Reproducible random number generation.

Use set_seed() at the top of every experiment script. For anything that
needs randomness inside a function (initial parameters, SPSA perturbations,
shadow snapshots), pass around the returned numpy Generator instead of
calling np.random directly. This keeps runs with the same seed identical.
"""

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """
    Seed Python, NumPy and (if installed) PyTorch.

    Returns a numpy Generator to be passed explicitly to functions that
    need random numbers.

    Note: PennyLane devices that sample shots take their own seed argument,
    e.g. qml.device("default.qubit", wires=n, shots=1000, seed=seed).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return np.random.default_rng(seed)


def child_seeds(master_seed: int, n: int) -> list:
    """
    Generate n statistically independent seeds from one master seed.

    Useful when a single experiment needs separate seeds for, say,
    each random VQE initialisation.
    """
    seq = np.random.SeedSequence(master_seed)
    return [int(s.generate_state(1)[0]) for s in seq.spawn(n)]
