import numpy as np
from compressed_mul import *

N_SAMPLES = 1000000


if __name__ == "__main__":
    rng = setup(seed=None)

    n = 6
    b = 2
    d = 1
    
    # matrix_A = random_sparse_matrix(n, 0.1, rng)
    # matrix_B = random_sparse_matrix(n, 0.1, rng)

    matrix_A = sparse_matrix_generator(n, 0.7, rng)
    matrix_B = sparse_matrix_generator(n, 0.7, rng)


    # matrix_A = rng.uniform(0,1,(n,n))
    # matrix_B = rng.uniform(0,1,(n,n))

    print(matrix_A, "\n")
    print(matrix_B)


    result = np.matmul(matrix_A, matrix_B)
    bound = (np.linalg.norm(result, "fro")**2) / b

    mat = lambda : calculate_result(matrix_A, matrix_B, b, d, n, rng)
   
    samples = []
    prog = 0

    for i in range(N_SAMPLES):
        if i % (N_SAMPLES / 100) == 0:
            prog += 1
            print(f"{prog}% done")
        samples.append(mat())
    
    samples = np.asarray(samples)
    variance = np.var(samples, axis=0, ddof=1)
    print(variance.sum())
    variance_list = variance.flatten().tolist()
    # print(bound)
    
    ones = [1 if v <= bound else 0 for v in variance_list]
    print(np.array(ones).reshape((n,n)))
    print("Number of 1s:", sum(ones), "of", len(variance_list))
    print(all([v <= bound for v in variance_list]))
    
    

