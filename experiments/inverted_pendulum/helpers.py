import matplotlib.pyplot as plt
import numpy as np


def clamp(number, min, max):
    if number < min:
        return min
    elif number > max:
        return max
    else:
        return number
