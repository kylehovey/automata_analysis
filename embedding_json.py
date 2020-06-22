from tqdm import tqdm
import numpy as np
import json

if __name__ == "__main__":
    out_file = './all_rules_embedding.json'
    embedding_file = './all_rules_embedded.npy'
    embedding = np.load(embedding_file)
    target_file = './target.npz'
    target = np.load('./target.npz')

    print (embedding.max(axis=0))
    print (embedding.min(axis=0))

    out = [{ "rule": int(rule), "embedding": list(map(lambda x: round(float(x), 3), location)) } for rule, location in zip(target, embedding)]

    with open(out_file, 'w') as out_file:
        json.dump(out, out_file)
