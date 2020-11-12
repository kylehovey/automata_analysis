from random import randrange
from tqdm import tqdm
import numpy as np
import os
import json

# { loc: [a,b], diff: 1, rule: "" }

if __name__ == "__main__":
    data = np.load('./rule_data.npz')
    diffs = np.load('./rule_diff.npz')
    targets = np.load('./target.npz')

    assert(len(data) == len(targets))

    N = len(data)

    a = randrange(0, N)
    b = randrange(0, N)

    #a = 148790
    #b = 237888
    a = 131070
    b = 238566

    print("Random axes are {} and {}".format(a, b))

    primary = data[a]
    secondary = data[b]

    primary_hat = primary / np.linalg.norm(primary)
    secondary_hat = secondary / np.linalg.norm(secondary)
    dot_product = np.dot(primary_hat, secondary_hat)
    angle = np.arccos(dot_product)

    print("Angle: {}".format(np.degrees(angle)))

    out = []
    scale = 100000.0

    for complexity, diff, rule_number in tqdm(list(zip(data, diffs, targets))):
        u = np.dot(primary, complexity)
        v = np.dot(secondary, complexity)

        out.append({
            "loc": [ float(u / scale), float(v / scale) ],
            "diff": float(diff.mean()),
            "rule": int(rule_number),
        })

    with open("random_embedding.json", "w") as out_file:
        json.dump(out, out_file)

