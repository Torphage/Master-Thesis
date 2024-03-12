#pragma once
#ifndef HASHING_HPP
#define HASHING_HPP

#include "utils.hpp"

#include <Eigen/Dense>
#include <random>
#include <iostream>


/**
 * @brief 
 * 
 * @tparam T 
 */
template <typename T>
struct Hashes {
    T h1;
    T h2;
    T s1;
    T s2;
};

/**
 * @brief 
 */
class FullyRandomHash {
  public:
    MatrixXui h1;
    MatrixXui h2;
    MatrixXui s1;
    MatrixXui s2;
    /**
     * @brief Constructs a \p Hashes containing the coefficients for \p d fully 
     *        random hash functions for each of h1, h2, s1, s2  
     * 
     * @param n is the size range of values that are hashed
     * @param b is the range to which values are hashed
     * @param d is the number of hash functions that will be used
     * @param seed 
     */
    FullyRandomHash(int n, int b, int d, int seed) {
        h1.resize(d, n);
        h2.resize(d, n);
        s1.resize(d, n);
        s2.resize(d, n);

        std::mt19937_64 rng(seed);

        std::uniform_int_distribution<uint64_t> bi(0, b - 1);

        for (int i = 0; i < d * n; i++) {
            h1.data()[i] = bi(rng);
            h2.data()[i] = bi(rng);
            s1.data()[i] = rng() % 2;
            s2.data()[i] = rng() % 2;
        }
    };

    /**
     * @brief A fully random hash function
     * 
     * @param map is the hash matrix (each contains the coefficients for a unique hash function)
     * @param index is the specific hash function (out of d total) to be used
     * @param x is the key to be hashed
     * @param range is the *unused* range to which values are hashed
     * @return int The hashed value
     */
    int operator()(MatrixXui &coeffs, int index, uint32_t x, int) {
        return coeffs(index, x);
    }
};

/**
 * @brief 
 * 
 */
class MultiplyShiftHash {
  public:
    MatrixXui h1;
    MatrixXui h2;
    MatrixXui s1;
    MatrixXui s2;
    /**
     * @brief Construct a new Multiply Shift Hash object
     * 
     * @param d 
     * @param seed 
     */
    MultiplyShiftHash(int d, int seed) {
        h1.resize(d, 2);
        h2.resize(d, 2);
        s1.resize(d, 2);
        s2.resize(d, 2);

        std::mt19937_64 rng(seed);

        std::uniform_int_distribution<uint64_t> uni(0, UINT64_MAX);

        for (int i = 0; i < 2 * d; i++) {
            h1.data()[i] = uni(rng);
            h2.data()[i] = uni(rng);
            s1.data()[i] = uni(rng);
            s2.data()[i] = uni(rng);
        }
    };

    /**
     * @brief A multiply-shift hash function
     * 
     * @param map is the hash matrix (each contains the coefficients for a unique hash function)
     * @param index is the specific hash function (out of d total) to be used
     * @param x is the key to be hashed
     * @param range is the range to which values are hashed
     * @return int The hashed value
     */
    int operator()(MatrixXui &coeffs, int index, uint32_t x, int range) {
        uint64_t u = coeffs(index, 0);
        uint64_t v = coeffs(index, 1);

        return (range * ((u * x + v) >> 32)) >> 32;
    }
};

class TabulationHash {
  private:
    int r;
    int t;

  public:
    MatrixXui h1;
    MatrixXui h2;
    MatrixXui s1;
    MatrixXui s2;
    
    /**
     * @brief Constructs a \p Hashes containing the coefficients for \p d tabulation
     *        hash functions for each of h1, h2, s1, s2  
     * @param p is the number of bits in the key to be hashed
     * @param q is the number of bits of the values that are hashed to 
     * @param r is the block size, i.e the number of blocks that the key is divided into
     * @param d is the number of hash functions that will be used
     * @param seed 
     */
    TabulationHash(int p, uint64_t q, int r, int d, int seed) : r(r) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> uni(0, static_cast<uint64_t>(1) << q);

        int size = 1 << r;
        t = ceil(p / r);

        h1.resize(t * d, size);
        h2.resize(t * d, size);
        s1.resize(t * d, size);
        s2.resize(t * d, size);

        for (int i = 0; i < t * d; i++) {
            for (int j = 0; j < size; j++) {
                h2(i, j) = uni(rng);
                h1(i, j) = uni(rng);
                s1(i, j) = uni(rng);
                s2(i, j) = uni(rng);
            }
        }
    };

    /**
     * @brief A tabulation hash function
     * 
     * @param map is a vector of tabulation matrices (represented as a matrix, each entry represents a unique hash function)
     * @param index is the specific hash function (out of d total) to be used
     * @param x is the key to be hashed
     * @param range is the range to which values are hashed
     * @return int The hashed value
     */
    int operator()(MatrixXui &coeffs, int index, uint32_t x, int range) {
        uint32_t res = 0;
        uint32_t mask = (1 << r) - 1;

        for (int i = 0; i < t; i++) {
            res ^= coeffs(t * index + i, (x >> r * i) & mask);
        }

        return res % range;
    }
};

#endif
