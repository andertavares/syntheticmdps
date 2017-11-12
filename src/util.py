import numpy as np


def meeting_point(a, b, window=100, start=0):
    """
    Determines the point where the moving average of a meets that of b

    """
    cva = np.convolve(a, np.ones((window,))/window, mode='valid')
    cvb = np.convolve(b, np.ones((window,))/window, mode='valid')

    for x, (val_a, val_b) in enumerate(zip(cva, cvb)):

        if x > start and val_a > val_b:
            return x
    return -1
