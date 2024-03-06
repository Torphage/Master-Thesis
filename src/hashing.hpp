#pragma once
#ifndef HASHING_HPP
#define HASHING_HPP

#include <Eigen/Dense>
#include <random>

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXui;

/**
 * @brief A struct containing the coefficients for the sets of hash functions h1, h2, s1, s2
 * 
 * @tparam T is the type that contains the coeffients
 */
template <typename T>
struct Hashes {
    T h1;
    T h2;
    T s1;
    T s2;
};


/**
 * @brief Constructs a \p Hashes containing the coefficients for \p d fully 
 *        random hash functions for each of h1, h2, s1, s2  
 * 
 * @param n is the size range of values that are hashed
 * @param b is the range to which values are hashed
 * @param d is the number of hash functions that will be used
 * @param rng is the random number generator
 * @return Hashes<Eigen::MatrixXi> 
 */
Hashes<Eigen::MatrixXi> fully_random_constructor(int n, int b, int d, std::mt19937_64 &rng);

/**
 * @brief A functor representing the fully random hash function. This will help
 *        the function to be inline.
 */
struct fully_random_hash {
    /**
     * @brief A fully random hash function
     * 
     * @param map is the hash matrix (each contains the coefficients for a unique hash function)
     * @param index is the specific hash function (out of d total) to be used
     * @param x is the key to be hashed
     * @param sign is the indicator for the function to hash to the values {-1,1}
     * @return int The hashed value
     */
    int operator()(Eigen::MatrixXi &map, int index, uint32_t x, int) {
        return map(index, x);
    }
};

/**
 * @brief Constructs a \p Hashes containing the coefficients for \p d multiply-shift 
 *        hash functions for each of h1, h2, s1, s2  
 * 
 * @param d is the number of hash functions that will be used
 * @param rng is the random number generator
 * @return Hashes<Eigen::MatrixXui> 
 */
Hashes<MatrixXui> multiply_shift_constructor(int d, std::mt19937_64 &rng);

/**
 * @brief A functor representing the multiply-shift hash function. This will help
 *        the function to be inline.
 */
struct multiply_shift_hash {
    /**
     * @brief A multiply-shift hash function
     * 
     * @param map is the hash matrix (each contains the coefficients for a unique hash function)
     * @param index is the specific hash function (out of d total) to be used
     * @param x is the key to be hashed
     * @param sign is the indicator for the function to hash to the values {-1,1}
     * @param b is the range to which values are hashed
     * @return int The hashed value
     */
    int operator()(MatrixXui &map, int index, uint32_t x, int sign, int b) {
        uint64_t u = map(index, 0);
        uint64_t v = map(index, 1);

        uint64_t intermediate = (u * x + v) >> 32;

        uint64_t product;
        if (sign) {
            product = intermediate * 2;
            return static_cast<int>(2 * (product >> 32)) - 1;
        } else {
            product = intermediate * b;
            return product >> 32;
        }
    }
};

/**
 * @brief Constructs a \p Hashes containing the coefficients for \p d tabulation
 *        hash functions for each of h1, h2, s1, s2  
 * @param p is the number of bits in the key to be hashed
 * @param q is the number of bits of the values that are hashed to 
 * @param r is the block size, i.e the number of blocks that the key is divided into
 * @param d is the number of hash functions that will be used
 * @param rng is the random number generator
 * @return Hashes<std::vector<MatrixXui>> 
 */
Hashes<std::vector<MatrixXui>> tabulation_constructor(int p, int q, int r, int d, std::mt19937_64 &rng);

/**
 * @brief A functor representing the tabulation hash function. This will help
 *        the function to be inline.
 */
struct tabulation_hash {
    /**
     * @brief A tabulation hash function
     * 
     * @param map is a vector of tabulation matrices (each entry represents a unique hash function)
     * @param index is the specific hash function (out of d total) to be used
     * @param x is the key to be hashed
     * @param sign is the indicator for the function to hash to the values {-1,1}
     * @param b is the range to which values are hashed
     * @param r is the block size, i.e the number of blocks that the key is divided into
     * @param t is the 
     * @return int The hashed value
     */
    int operator()(std::vector<MatrixXui> &map, int index, uint32_t x, int sign, int b, uint32_t r, int t) {
        uint32_t res = 0;
        MatrixXui &tab_matrix = map[index];
        uint32_t mask = (1 << r) - 1;

        for (int i = 0; i < t; i++) {
            res ^= tab_matrix(i, (x >> r * i) & mask);
        }

        if (sign) {
            return 2 * static_cast<int>(res % 2) - 1;
        } else {
            return res % b;
        }
    }
};

#endif
