import multiprocessing as mp
import numpy as np
import scipy.signal


def get_cpu_count():
    return mp.cpu_count()


def is_prime(n):
    for divisor in range(2, int(n ** 0.5) + 1):
        if n % divisor == 0:
            return 0
    return 1

def inefficient_fib(n=None):
    """Compute intensive calculation for the nth fibonacci number"""
    if n <= 1:
        return n
    return inefficient_fib(n - 1) + inefficient_fib(n - 2)

iterations_count = iterations_count = round(1e4)
def complex_operation_numpy(index):
    data = np.ones(iterations_count)
    val = np.exp(data) * np.sinh(data)
    return val.sum()

def f_ray_image_signal(args):
    image, random_filter = args
    # Do some image processing: convolve two 2-dimensional arrays.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

def f_py_image_signal(args):
    image, random_filter = args
    # Do some image processing: convolve two 2-dimensional arrays.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

