from tqdm import tqdm
import numpy as np
import os
import json

if __name__ == "__main__":
    data = np.load('./rule_data.npz')
    targets = np.load('./target.npz')

    assert(len(data) == len(targets))

    N = len(data)
    min_dot = 1e20
    min_pair = None

    everything = list(
        zip(
            data,
            targets
        )
    )

    with tqdm(total=N*N) as bar:
        for cA, rA in everything:
            for cB, rB in everything:
                bar.update(1)
                product = np.dot(cA, cB)
                if product < min_dot:
                    min_dot = product
                    min_pair = [rA, rB]

    print(max_dist)
    print(max_pair)
