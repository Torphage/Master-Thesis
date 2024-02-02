import numpy as np
import sympy

    
def rand_primes(low, d):

    primes = [sympy.randprime(low, 2**61) for _ in range(d)]

    np.array(primes).reshape(d,1)



def gen_coeffs(n,k,rng):

    a = rng.integers(1, 2**61, (n,1))
    b = rng.integers(0, 2**61, (n,k-1))

    return np.concatenate((a, b), axis=-1)


def poly_k_indendent_hash(key, coeffs, p, b):

    sum = 0

    for i, coeff in enumerate(coeffs):
        sum += ((key**i) * coeff)
    
    return (sum % p) % b


def tab_k_independent_hash():

    pass