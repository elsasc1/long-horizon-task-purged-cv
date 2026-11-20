"""Deterministic corpora for the hidden tests (verifier-owned)."""
import numpy as np

LETTERS = np.array(list("abcdefghijklmnopqrstuvwxyz"))


class _Words:
    """Pool of fresh random words (so that most shingles are rare) + a small common vocabulary."""

    def __init__(self, rng, n_fresh):
        L = rng.integers(4, 10, size=n_fresh)
        chars = rng.integers(97, 123, size=(n_fresh, 9)).astype(np.uint8)
        self.fresh = [chars[i, : L[i]].tobytes().decode() for i in range(n_fresh)]
        self.common = [self.fresh[i] for i in range(200)]
        self.pos = 200
        self.rng = rng

    def word(self):
        if self.rng.random() < 0.15:
            return self.common[int(self.rng.integers(0, 200))]
        w = self.fresh[self.pos]
        self.pos += 1
        return w


def boundary_pairs(t_pairs, offset=0):
    """Pairs of texts built from distinct CJK characters whose shingle sets have an EXACT Jaccard of b/a.

    X has L chars -> a = L-4 shingles; Y is the first k chars of X -> b = k-4 shingles, all shared.
    So J = b / a exactly. Also returns the 'just below' partner (one shingle fewer).
    """
    texts, expect = [], []   # expect[i] = list of (index_in_texts_a, index_b, linked_bool_for_threshold)
    cur = 0x4E00 + offset
    for (a, b, linked_at) in t_pairs:
        L = a + 4
        x = "".join(chr(cur + i) for i in range(L)); cur += L
        y = x[: b + 4]
        texts += [x, y]
        expect.append((len(texts) - 2, len(texts) - 1, linked_at))
    return texts, expect


def make_corpus(n=10_000, seed=0, with_boundary_at=0.7):
    rng = np.random.default_rng(seed)
    W = _Words(rng, n * 70 + 1000)
    headers = [" ".join(W.word() for _ in range(14)) for _ in range(5)]
    footer = "all rights reserved see terms and conditions"

    def body():
        return [W.word() for _ in range(30)]

    def render(h, words):
        return headers[h] + " " + " ".join(words) + " " + footer

    def mutate(words, k):
        w = list(words)
        for pos in rng.choice(len(w), size=k, replace=False):
            w[pos] = W.word()
        return w

    texts = []
    while len(texts) < n:
        h = int(rng.integers(0, 5))
        base = body()
        texts.append(render(h, base))
        if rng.random() < 0.45:
            cur = base
            for _ in range(int(rng.integers(1, 6))):
                src = cur if rng.random() < 0.4 else base      # chain vs star
                cur = mutate(src, int(rng.integers(1, 7)))
                if rng.random() < 0.5:
                    texts.append(render(h, cur).upper())          # case-only noise on top
                else:
                    texts.append("  " + render(h, cur).replace(" ", "  ", 3))   # whitespace noise
    texts = texts[:n]
    if with_boundary_at is not None:
        pairs = [(10, 7, True), (20, 14, True), (30, 21, True), (40, 28, True), (50, 35, True),
                 (60, 42, True), (70, 49, True), (10, 6, False), (20, 13, False), (30, 20, False)]
        bt, _ = boundary_pairs(pairs)
        texts += bt
    perm = rng.permutation(len(texts))
    return [texts[i] for i in perm]
