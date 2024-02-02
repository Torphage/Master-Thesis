import numpy as np
from compressed_mul import *



if __name__ == "__main__":
    rng = setup()

    n = 10
    b = 10
    d = 1
    
    matrix_A = random_sparse_matrix(n, 0.1, rng)
    matrix_B = random_sparse_matrix(n, 0.1, rng)

    result = np.matmul(matrix_A, matrix_B)
    bound = np.linalg.norm(result, "fro")**2 / b


    mat = lambda : calculate_result(matrix_A, matrix_B, b, d, n, rng)
    samples = [mat() for _ in range(100)]
    # samples = []
    # for i in range(100):
    #     mat = calculate_result(matrix_A, matrix_B, b, d, n, rng)
    #     samples.append(mat)
    #     print(i)
    
    samples = np.asarray(samples)
    variance = np.var(samples, axis=0, ddof=1)
    print(variance.sum())
    variance_list = variance.flatten().tolist()
    # print(bound)
    
    ones = [1 if v <= bound else 0 for v in variance_list]
    print(np.array(ones).reshape((n,n)))
    print("Number of 1s:", sum(ones), "of", len(variance_list))
    print(all([v <= bound for v in variance_list]))
    
    

