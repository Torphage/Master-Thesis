import numpy as np
import scipy as sp
from hashing import FullyRandomHash, MultiplyShiftHash, TabulationHash


def compressed_product(m1, m2, hashes):
    b = hashes.b
    d = hashes.d
    n, _ = m1.shape
    p = np.zeros((d, b), dtype="complex128")
    
    for t in range(d):
        for k in range(n):
            pa, pb = np.zeros(b), np.zeros(b)

            for i in range(n):
                pa[hashes.hash("h1", t, i)] += hashes.hash("s1", t, i) * m1[i, k]
                pb[hashes.hash("h2", t, i)] += hashes.hash("s2", t, i) * m2[k, i]

            pa = np.fft.rfft(pa)
            pb = np.fft.rfft(pb)
            for z in range(b//2 + 1):
                p[t, z] += pa[z] * pb[z]
    
    p = np.fft.irfft(p, b, axis=-1)
    
    return p


def decompress_element(p, i, j, hashes):
    d = hashes.d
    b = hashes.b
    xt = np.zeros(d)
    for t in range(d):
        a1 = hashes.hash("s1", t, i)
        a2 = hashes.hash("s2", t, j)
        a3 = hashes.hash("h1", t, i)
        a4 = hashes.hash("h2", t, j)
        xt[t] = a1 * a2 * np.real(p[t, int((a3 + a4) % b)])
    return np.median(xt)


def decompress_matrix(p, n, hashes):
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = decompress_element(p, i, j, hashes)

    return c


def calculate_result(m1, m2, n, hashes, rng):
    p = compressed_product(m1, m2, hashes)
    return decompress_matrix(p, n, hashes)


def random_sparse_matrix(n, density, rng):
    """
    Generates a random sparse n×n matrix with the
    option to specify the density of non-zero elements.
    """
    return sp.sparse.random(n, n, density=density, random_state=rng, dtype = "float64").A


def sparse_matrix_generator(n, density, rng):
   uni = rng.uniform(0, 1, (n, n))
   indices = np.random.choice(range(n*n), int(n*n*(1-density)), replace=False)
   uni = uni.flatten()
   for i in indices:
       uni[i] = 0
   uni = uni.reshape((n,n))
   return uni


def pretty_print_matrix(mat, n):
    """
    Pretty prints matrices such that any two matrices lines up when printing them
    """
    rounded_digit = 6
    if n > 31:
        # These are the parts of the matrix that will be displayed
        q1 = mat[:3, :3]  # top left
        q2 = mat[:3, n - 3 :]  # top right
        q3 = mat[n - 3 :, :3]  # bottom left
        q4 = mat[n - 3 :, n - 3 :]  # bottom right

        # if any displayed digits are negative, then round the matrix by of less
        if np.any((q1 < 0) | (q2 < 0) | (q3 < 0) | (q4 < 0)):
            rounded_digit = 5
    else:
        if np.any(mat < 0):
            rounded_digit = 5

    print(mat.round(rounded_digit))


def setup(seed=None):
    # Disables scientific notation for floats
    np.set_printoptions(suppress=True)

    if not seed:
        seed = int(np.random.rand() * (2**32 - 1))
        print(f"seed: {seed}")
    return np.random.default_rng(seed=seed)


if __name__ == "__main__":
    seed = 2
    rng = setup(seed = seed)  # Set seed here to reproduce results

    # user variables
    n = 85  # The size of the matrix
    b = 2500 # Number of buckets
    d = 24 # Number of hash functions

    matrix_A = random_sparse_matrix(n, 0.05, rng)  # Can also hard code a matrix here
    matrix_B = random_sparse_matrix(n, 0.05, rng)  # Can also hard code a matrix here

    # matrix_A = rng.uniform(0,1,(n,n))
    # matrix_B = rng.uniform(0,1,(n,n))
    
    # hashes = FullyRandomHash(n, d, b, rng)
    # hashes = MultiplyShiftHash(d, b, rng)
    r, p, q = 8, 32, 32
    hashes = TabulationHash(r, p, q, d, b, rng)

    print(f"seed={seed}   n={n}   b={b}   d={d}")
    if hashes.__class__.__name__ == "TabulationHash":
        print(f"{hashes.__class__.__name__}   r={r}   p={p}   q={q}")
    else:
        print(hashes.__class__.__name__)
    
    
    # Calculate the final matrix product
    p = compressed_product(matrix_A, matrix_B, hashes)
    result = decompress_matrix(p, n, hashes)

    # This rounds all the elements in the result matrix to the 15th decimal.
    # This is used to offset Python's floating point precision problem.
    # It's used when counting the number of non-zero elements.
    rounded_res = result.round(12)
    print("\n------ Compressed Matrix Product ------")
    print(f"Non-zero elements: {np.count_nonzero(rounded_res)} of {n*n}")
    print(f"Sum of elements: {np.sum(result)}")
    rounded_res[rounded_res == 0.0] = 0.0  # Converts -0 to 0 (they're equivalent)
    print("Product:")
    pretty_print_matrix(rounded_res, n)

    print("\n--------- Real Matrix Product ---------")
    real_product = np.matmul(matrix_A, matrix_B)
    r = real_product.round(12)
    print(f"Non-zero elements: {np.count_nonzero(r)} of {n*n}")
    print(f"Sum of elements: {np.sum(real_product)}")
    r[r == 0.0] = 0.0  # Converts -0 to 0 (they're equivalent)
    print("Product:")
    pretty_print_matrix(r, n)
