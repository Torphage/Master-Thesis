import numpy as np
from abc import ABC, abstractmethod


class BaseHash(ABC):
    def __init__(self, d, b, rng):
        self.d = d
        self.b = b
        self.rng = rng

    @abstractmethod
    def hash(self, name, index, x):
        return NotImplemented

class FullyRandomHash(BaseHash):
    def __init__(self, n, d, b, rng):
        super().__init__(d, b, rng)
        self.fun = {
            "h1": rng.integers(0, b, (d, n)),
            "h2": rng.integers(0, b, (d, n)),
            "s1": rng.choice([-1, 1], (d, n)),
            "s2": rng.choice([-1, 1], (d, n)),
        }
    
    def hash(self, name, index , x):
        return self.fun[name][index, x]
        

class MultiplyShiftHash(BaseHash):
    def __init__(self, d, b, rng):
        super().__init__(d, b, rng)
        self.fun = {
            "h1": rng.integers(0, 2**64, (d, 2), dtype = "uint64"),
            "h2": rng.integers(0, 2**64, (d, 2), dtype = "uint64"),
            "s1": rng.integers(0, 2**64, (d, 2), dtype = "uint64"),
            "s2": rng.integers(0, 2**64, (d, 2), dtype = "uint64"),
        }
        
    def hash(self, name, index, x):
        u = self.fun[name][index][0]
        v = self.fun[name][index][1]
        product1 = np.multiply(u, x, dtype="uint64")
        sum1 = np.add(product1, v, dtype="uint64")
        intermediate = np.right_shift(sum1, 32, dtype = "uint64")
        
        if (name[0] == "s"):
            product2 = np.multiply(intermediate, 2, dtype="uint64")
            return 2 * np.right_shift(product2, 32, dtype="uint64") - 1
        else:
            product2 = np.multiply(intermediate, self.b, dtype="uint64")
            return np.right_shift(product2, 32, dtype="uint64")
 

class TabulationHash(BaseHash):
    def __init__(self, r, p, q, d, b, rng):
        super().__init__(d, b, rng)
        self.r = r
        self.p = p
        self.q = q
        self.t = int(np.ceil(p/r))
        self.fun = {
            key: np.asarray([
                rng.integers(0, 2**self.q, (self.t, 2**self.r), dtype = "uint64") 
                for _ in range(d)
            ]) 
            for key in ["h1", "h2", "s1", "s2"]
        }
    
    def hash(self, name, index, x):
        res = np.uint64(0)
        mask = (1 << self.r) - 1
        tab_matrix = self.fun[name][index]

        for i in range(self.t):
            shift = np.right_shift(x, (self.r * i))
            masked = np.bitwise_and(shift, mask)
            res = np.bitwise_xor(res, tab_matrix[i][masked])

        if (name[0] == "s"):
            return 2 * np.mod(res, 2) - 1
        else:
            return np.mod(res, self.b, dtype = "uint64")