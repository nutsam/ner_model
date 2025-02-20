import logging
import signal


class TimeoutError(Exception):
    """Exception raised when a function execution exceeds the allotted time."""

    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


def run_with_timeout(func, args=(), kwargs=None, timeout_sec=5):
    """
    Executes a function with a timeout.

    Args:
        func (callable): Function to execute.
        args (tuple): Positional arguments for the function.
        kwargs (dict): Keyword arguments for the function.
        timeout_sec (int): Timeout in seconds.

    Returns:
        Any: The result of the function call.

    Raises:
        TimeoutError: If the execution exceeds the timeout.
    """
    if kwargs is None:
        kwargs = {}
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result
