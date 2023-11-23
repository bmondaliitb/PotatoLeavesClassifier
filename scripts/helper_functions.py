#!/usr/bin/env python3
import numpy as np

# Custom generator to combine both datasets
def combined_generator(gen1, gen2):
    while True:
        if np.random.choice([True, False]):  # Randomly choose which generator to yield from
            yield next(gen1)
        else:
            yield next(gen2)

