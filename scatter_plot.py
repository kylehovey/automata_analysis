import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# {'loc': [592.9939629999996, 245.16197000000142], 'diff': -5.435294117647059, 'rule': 0}

def load_json_from(file):
    with open(file) as infile:
        return json.load(infile)

def assert_in_order(embedding, label):
    print("Verifying order of {} embedding...".format(label))

    N = len(embedding)
    i = 0

    with tqdm(total=N-1) as bar:
        while i < N - 1:
            assert(embedding[i]["rule"] + 1 == embedding[i+1]["rule"])
            i += 1
            bar.update(1)

    print("Verified.")

def vectorize(embedding):
    x = [ data["loc"][0] for data in embedding ]
    y = [ data["loc"][1] for data in embedding ]
    diffs = [ data["diff"] for data in embedding ]
    rules = [ data["rule"] for data in embedding ]

    return map(
        lambda arr: np.array(arr),
        (x, y, diffs, rules)
    )


def scatter_plot_for(embedding, filename):
    x = [ data["loc"][0] for data in embedding ]
    y = [ data["loc"][1] for data in embedding ]
    c = [ data["diff"] for data in embedding ]

    plt.scatter(x, y, c=c, s=2)

    plt.savefig(filename)

if __name__ == "__main__":
    print("Loading JSON embedding files...")

    raw_dots_data = load_json_from("./random_embedding.json")
    raw_umap_data = load_json_from("./embedding_umap.json")

    print("Loaded.")

    assert(len(raw_dots_data) == len(raw_umap_data))
    assert_in_order(raw_dots_data, "dot product")
    assert_in_order(raw_umap_data, "UMAP")

    xd, yd, diffsd, rulesd = vectorize(raw_dots_data)
    xu, yu, diffsu, rulesu = vectorize(raw_umap_data)

    xdn = xd / np.max(xd)
    ydn = yd / np.max(yd)
    xun = xu / np.max(xu)
    yun = yu / np.max(yu)

    N = len(xd)
    steps = 256
    step_number = 0
    dt = 1.0 / steps
    t = 0

    print("Saving images...")

    with tqdm(steps) as bar:
        while step_number <= steps:
            x = (1 - t) * xdn + t * xun
            y = (1 - t) * ydn + t * yun

            plt.scatter(x, y, c=diffsu, s=2)
            plt.savefig("./animate_embedding/step_{}.png".format(step_number))
            plt.clf()
            bar.update(1)

            step_number += 1
            t += dt
