import json
import pickle
import random
import os

random.seed(42)

def build_embeddings(train_path, out_path='word_embedding.pkl', dim=50):
    with open(train_path, encoding='utf-8') as f:
        data = json.load(f)
    vocab = set()
    for elt in data:
        if 'text' in elt:
            for w in elt['text'].split():
                vocab.add(w.lower())
    embeddings = {}
    for w in vocab:
        embeddings[w] = [random.uniform(-0.1,0.1) for _ in range(dim)]
    embeddings['unk'] = [random.uniform(-0.1,0.1) for _ in range(dim)]
    with open(out_path, 'wb') as out:
        pickle.dump(embeddings, out)
    print(f"Wrote {len(embeddings)} embeddings to {out_path}")

if __name__ == '__main__':
    import sys
    train = sys.argv[1] if len(sys.argv) > 1 else 'training.json'
    build_embeddings(train)
