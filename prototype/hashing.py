import numpy as np
import sympy

    
def rand_primes(low, d):

    primes = [sympy.randprime(low, 2**64-1) for _ in range(d)]

    return np.array(primes).reshape(d,1)



def gen_coeffs(size,rng):

    return rng.integers(0,  2**64-1, size, dtype = "uint64")


def random_hash(x, hashes):
    
    return hashes[x]

def multiply_shift_hash(x, coeffs, b):
    product1 = np.multiply(coeffs[0], np.uint32(x), dtype="uint64")
    intermediate = np.right_shift(product1 + coeffs[1], 32, dtype = "uint64")
    product2 = np.multiply(intermediate, np.uint32(b), dtype="uint64")
    return np.right_shift(product2, 32, dtype="uint64")

def tabulation_matrix(p, r, rng):

    # block size r
    # p number of bits in key to hash
    # q number of output bits
    # t = ceil(p/r)

    return rng.integers(0, 2**64-1, ((np.ceil(p/r)).astype("int64"), 2**r), dtype = "uint64")



def tabulation_hash(x, tab_matrix, r, b):

    res = np.uint64(0)

    for i in range(r):

        index = np.right_shift(x, r * i, dtype="uint64")
        res = np.bitwise_xor(res, tab_matrix[i][index])

    return np.mod(res, b, dtype = "uint64")
