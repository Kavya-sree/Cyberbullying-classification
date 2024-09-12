import numpy as np

def clr_triangular(epoch, step_size=2000, base_lr=1e-5, max_lr=1e-2):
    """
    Cyclical Learning Rate scheduler.
    """
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr
