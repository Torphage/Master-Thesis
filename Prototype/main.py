import numpy as np
import scipy as sp


def init_arrays(b, d, n, rng):
    # All the variables that are declared here are initiated as 2d matrices.
    p = np.zeros((d, b), dtype="complex_")
    h1, h2 = rng.integers(0, b, (d, n)), rng.integers(0, b, (d, n))
    s1, s2 = rng.choice([-1, 1], (d, n)), rng.choice([-1, 1], (d, n))
    return (p, h1, h2, s1, s2)


def compressed_product(m1, m2, b, d, n, rng):
    p, h1, h2, s1, s2 = init_arrays(b, d, n, rng)
    for t in range(d):
        for k in range(n):
            pa, pb = np.zeros(b, dtype="complex_"), np.zeros(b, dtype="complex_")

            for i in range(n):
                pa[h1[t, i]] += s1[t, i] * m1[i, k]
                pb[h2[t, i]] += s2[t, i] * m2[k, i]

            pa, pb = np.fft.fft(pa), np.fft.fft(pb)
            p[t] += np.multiply(pa, pb)

    p = np.fft.ifft(p, axis=-1)
    return (p, h1, h2, s1, s2)


def decompress_element(p, h1, h2, s1, s2, b, d, i, j):
    xt = np.zeros(d)
    for t in range(d):
        xt[t] = s1[t, i] * s2[t, j] * np.real(p[t, (h1[t, i] + h2[t, j]) % b])
    return np.median(xt)


def decompress_matrix(p, h1, h2, s1, s2, b, d, n):
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = decompress_element(p, h1, h2, s1, s2, b, d, i, j)
    return c


def calculate_result(m1, m2, b, d, n, rng):
    p, h1, h2, s1, s2 = compressed_product(m1, m2, b, d, n, rng)
    return decompress_matrix(p, h1, h2, s1, s2, b, d, n)


def random_sparse_matrix(n, density, rng):
    """
    Generates a random sparse n×n matrix with the
    option to specify the density of non-zero elements.
    """
    return sp.sparse.random(n, n, density=density, random_state=rng).A


def pretty_print_matrix(mat, n):
    """
    Pretty prints matrices such that any two matrices lines up when printing them
    """
    rounded_digit = 3
    if n > 31:
        # These are the parts of the matrix that will be displayed
        q1 = mat[:3, :3]  # top left
        q2 = mat[:3, n - 3 :]  # top right
        q3 = mat[n - 3 :, :3]  # bottom left
        q4 = mat[n - 3 :, n - 3 :]  # bottom right

        # if any displayed digits are negative, then round the matrix by of less
        if np.any((q1 < 0) | (q2 < 0) | (q3 < 0) | (q4 < 0)):
            rounded_digit = 2
    else:
        if np.any(mat < 0):
            rounded_digit = 2

    print(mat.round(rounded_digit))


def setup(seed=None):
    # Disables scientific notation for floats
    np.set_printoptions(suppress=True)

    if not seed:
        seed = int(np.random.rand() * (2**32 - 1))
        print(f"seed: {seed}")
    return np.random.default_rng(seed=seed)


if __name__ == "__main__":
    rng = setup(seed=1337)  # Set seed here to reproduce results

    # user variables
    n = 100  # The size of the matrix
    matrix_A = random_sparse_matrix(n, 0.05, rng)  # Can also hard code a matrix here
    matrix_B = random_sparse_matrix(n, 0.05, rng)  # Can also hard code a matrix here
    b = 2000
    d = 28

    # Calculate the final matrix product
    p, h1, h2, s1, s2 = compressed_product(matrix_A, matrix_B, b, d, n, rng)
    result = decompress_matrix(p, h1, h2, s1, s2, b, d, n)

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
