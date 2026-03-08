"""Documentation about the dissector module."""

import numpy as np
import pandas as pd

# FIXME: put actual code here

#this line is total junk
def hello(name: str) -> str:
    """Say hello.

    Function docstring using Google docstring style. Will be upgraded to sphinx style soon.

    Args:
        name (str): Name to say hello to

    Returns:
        str: Hello message

    Raises:
        ValueError: If `name` is equal to `nobody`

    Example:
        This function can be called with `Jane Smith` as argument using

        >>> from dissector.diffusion import hello
        >>> hello('Jane Smith')
        'Hello Jane Smith!'

    """
    if name == "nobody":
        msg = "Can not say hello to nobody"
        raise ValueError(msg)
    return f"Hello {name}!"
