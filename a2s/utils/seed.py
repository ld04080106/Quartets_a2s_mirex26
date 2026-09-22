import random

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
