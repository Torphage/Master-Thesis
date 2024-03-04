#pragma once
#ifndef HASHING_HPP
#define HASHING_HPP

#include <Eigen/Dense>
#include <random>
#include <unordered_map>

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXui;

/**
 * @brief
 *
 */
class BaseHash {
   protected:
    std::mt19937_64 rng;

   public:
    int b;
    int d;
    BaseHash(int b, int d, std::mt19937_64 &rng);
    virtual ~BaseHash() = default;
    /**
     * @brief
     *
     * @param name
     * @param index
     * @param x
     * @return int
     */
    virtual int hash(int name, int index, uint32_t x) = 0;
};

/**
 * @brief
 *
 */
class FullyRandomHash : public BaseHash {
   public:
    /**
     * @brief A hash-map, containing the hashing methods, with strings as keys.
     * Acceptable keys are `"h1"`, `"h2"`, `"s1"` and `"s2"`
     */
    std::array<Eigen::MatrixXi, 4> map;
    FullyRandomHash(int n, int b, int d, std::mt19937_64 &rng);
    int hash(int name, int index, uint32_t x);
};

/**
 * @brief
 *
 */
class MultiplyShiftHash : public BaseHash {
   public:
    /**
     * @brief A hash-map, containing the hashing methods, with strings as keys.
     * Acceptable keys are `"h1"`, `"h2"`, `"s1"` and `"s2"`
     */
    std::array<MatrixXui, 4> map;
    MultiplyShiftHash(int b, int d, std::mt19937_64 &rng);
    int hash(int name, int index, uint32_t x);
};

/**
 * @brief
 *
 */
class TabulationHash : public BaseHash {
    int r;
    int t;

   public:
    /**
     * @brief A vector of hash-maps, containing the hashing methods, with strings as keys.
     * Acceptable keys are `"h1"`, `"h2"`, `"s1"` and `"s2"`
     */
    std::array<std::vector<MatrixXui>, 4> map;
    TabulationHash(int p, int q, int r, int b, int d, std::mt19937_64 &rng);
    int hash(int name, int index, uint32_t x);
};

template <typename T>
struct Hashes {
    T h1;
    T h2;
    T s1;
    T s2;
};

Hashes<Eigen::MatrixXi> fully_random_constructor(int n, int b, int d, std::mt19937_64 &rng);

struct fully_random_hash {
    int operator()(Eigen::MatrixXi &map, int index, uint32_t x, int sign) {
        return map(index, x);
    }
};

Hashes<MatrixXui> multiply_shift_constructor(int d, std::mt19937_64 &rng);

struct multiply_shift_hash {
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

Hashes<std::vector<MatrixXui>> tabulation_constructor(int p, int q, int r, int d, std::mt19937_64 &rng);

struct tabulation_hash {
    int operator()(std::vector<MatrixXui> &map, int index, uint32_t x, int sign, int b, int r, int t) {
        uint32_t res = 0;
        MatrixXui &tab_matrix = map[index];
        int mask = (1 << r) - 1;

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
