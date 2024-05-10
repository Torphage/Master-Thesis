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
 *
 * @tparam Word
 */
template <typename Word>
class FullyRandomHash {
   private:
    typedef Eigen::Matrix<Word, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;

   public:
    MatrixType h1;
    MatrixType h2;
    MatrixType s1;
    MatrixType s2;
    /**
     * @brief Constructs a \p Hashes containing the coefficients for \p d fully
     *        random hash functions for each of h1, h2, s1, s2
     *
     * @param n is the size range of values that are hashed
     * @param b is the range to which values are hashed
     * @param d is the number of hash functions that will be used
     * @param seed
     */
    FullyRandomHash(const int n, const int b, const int d, int seed) {
        h1.resize(d, n);
        h2.resize(d, n);
        s1.resize(d, n);
        s2.resize(d, n);

        std::mt19937_64 rng(seed);

        std::uniform_int_distribution<Word> bi(0, b - 1);

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
     * @return Word The hashed value
     */
    Word operator()(MatrixType& coeffs, int index, int x, const int) const {
        return coeffs(index, x);
    }
};

/**
 * @brief
 *
 * @tparam Word
 * @tparam SmallWord
 */
template <typename Word, typename SmallWord>
class MultiplyShiftHash {
   private:
    typedef Eigen::Array<Word, Eigen::Dynamic, 2> MatrixType;
    static constexpr int size = 8 * sizeof(SmallWord);

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatrixType h1;
    MatrixType h2;
    MatrixType s1;
    MatrixType s2;
    /**
     * @brief Construct a new Multiply Shift Hash object
     *
     * @param d
     * @param seed
     */
    MultiplyShiftHash(const int d, int seed) {
        h1.resize(d, 2);
        h2.resize(d, 2);
        s1.resize(d, 2);
        s2.resize(d, 2);

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<Word> uni(0, std::numeric_limits<Word>::max());

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
     * @return SmallWord The hashed value
     */
    SmallWord operator()(MatrixType& coeffs, int index, SmallWord x, const SmallWord range) const {
        return (range * ((coeffs(index, 0) * x + coeffs(index, 1)) >> size)) >> size;
    }
};

/**
 * @brief A constexpr ceil function with the sole purpose of speeding up tabulation hashing
 *
 * @param dividend is the divident
 * @param divisor is the divisor
 * @return constexpr int
 */
constexpr int constexpr_ceil(int dividend, int divisor) {
    return (dividend + divisor - 1) / divisor;
}

/**
 * @brief
 *
 * @tparam WordIn is the number of bits in the key to be hashed
 * @tparam WordOut is the number of bits of the values that are hashed to
 * @tparam r is the block size, i.e the number of blocks that the key is divided into
 */
template <typename WordIn, typename WordOut, int r>
class TabulationHash {
   private:
    typedef Eigen::Matrix<WordOut, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
    static constexpr int p = 8 * sizeof(WordIn);
    static constexpr int q = 8 * sizeof(WordOut);
    static constexpr int t = constexpr_ceil(8 * sizeof(WordIn), r);
    // int t;

   public:
    MatrixType h1;
    MatrixType h2;
    MatrixType s1;
    MatrixType s2;

    /**
     * @brief Constructs a \p Hashes containing the coefficients for \p d tabulation
     *        hash functions for each of h1, h2, s1, s2
     * @param d is the number of hash functions that will be used
     * @param seed
     */
    TabulationHash(const int d, int seed) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<WordOut> uni(0, std::numeric_limits<WordOut>::max());

        int size = 1 << r;

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
     * @return WordOut The hashed value
     */
    WordOut operator()(MatrixType& coeffs, int index, int x, const int range) const {
        WordOut res = 0;
        constexpr WordOut mask = (1 << r) - 1;

        for (int i = 0; i < t; i++) {
            res ^= coeffs(t * index + i, (x >> r * i) & mask);
        }

        return res % range;
    }
};

#endif
