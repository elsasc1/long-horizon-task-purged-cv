"""Offline helper: brute-force ground truth for the scale corpus -> expected_scale.json.
Run once by the author (about 5 minutes on 2 cores); the result is committed next to the tests."""
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import datagen, oracle

N, SEED, T = 10_000, 0, 0.7   # N base texts + 20 boundary texts
texts = datagen.make_corpus(N, SEED, with_boundary_at=T)
t0 = time.time()
lab = oracle.clusters(texts, T, processes=2)
out = {
    "n": len(texts), "seed": SEED, "threshold": T, "n_clusters": max(lab) + 1,
    "labels_sha256": hashlib.sha256(",".join(map(str, lab)).encode()).hexdigest(),
    "largest_cluster": max(__import__("collections").Counter(lab).values()),
    "oracle_seconds": round(time.time() - t0, 1),
}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "expected_scale.json"), "w"), indent=1)
print(out)
