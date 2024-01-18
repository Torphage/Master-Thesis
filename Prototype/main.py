import numpy as np
import numpy.typing as npt
import scipy as sp
from typing import NamedTuple


class Hashes(NamedTuple):
    h1: npt.NDArray
    h2: npt.NDArray
    s1: npt.NDArray
    s2: npt.NDArray


def init_arrays(n, b, d, rng):
    p = np.zeros((d, b), dtype="complex_")
    h1, h2 = rng.integers(0, b, (d, n)), rng.integers(0, b, (d, n))
    s1, s2 = rng.choice([-1, 1], (d, n)), rng.choice([-1, 1], (d, n))
    return (p, h1, h2, s1, s2)


def compressed_product(m1, m2, b, d, rng, n):
    p, h1, h2, s1, s2 = init_arrays(n, b, d, rng)
    for t in range(d):
        for k in range(n):
            pa, pb = np.zeros(b), np.zeros(b)
            for i in range(n):
                pa[h1[t, i]] += s1[t, i] * m1[i, k]
            for j in range(n):
                pb[h2[t, j]] += s2[t, j] * m2[k, j]
            pa, pb = np.fft.fft(pa), np.fft.fft(pb)

            for z in range(b):
                p[t, z] += pa[z] * pb[z] # Casting complex valu
    for t in range(d):
        p[t] = np.fft.ifft(p[t])
    return (p, h1, h2, s1, s2)


def recover_element(p, h1, h2, s1, s2, b, d, i, j):
    xt = np.zeros(d)
    for t in range(d):
        xt[t] = s1[t, i] * s2[t, j] * p[t, (h1[t, i] + h2[t, j]) % b]
    return np.median(xt)


def decompress_matrix(p, h1, h2, s1, s2, b, d, n):
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = recover_element(p, h1, h2, s1, s2, b, d, i, j)
    return c


def main(seed=None):
    n = 12
    d = 15
    b = 5

    random1 = sp.sparse.random(12, 12, density=0.05, random_state=seed).A
    random2 = sp.sparse.random(12, 12, density=0.05, random_state=seed).A

    rng = np.random.default_rng(seed=seed)

    # pretty_print_compressed((p,h1,h2,s1,s2))
    print((random1 * random2).round(3))
    print()
    p, h1, h2, s1, s2 = compressed_product(random1, random2, b, d, rng, n)

    # print(compressed_product(m1, m2, b, d, rng, n))
    return decompress_matrix(p, h1, h2, s1, s2, b, d, n)


def pretty_print_compressed(c):
    print("")
    print(f"p:  {c[0].round(3)}")
    print("")


if __name__ == "__main__":
    result = main(seed=1984)
    print(result.round(2))
