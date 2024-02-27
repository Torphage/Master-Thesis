#pragma once
#ifndef HASHING_HPP
#define HASHING_HPP

#include <random>
#include <Eigen/Dense>
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
    /**
     * @brief A hash-map, containing the hashing methods, with strings as keys.
     * Acceptable keys are `"h1"`, `"h2"`, `"s1"` and `"s2"`
     */
    // std::unordered_map<std::string, MatrixXui> map;
    std::array<MatrixXui, 4> map;
  public:
    FullyRandomHash(int n, int b, int d, std::mt19937_64 &rng);
    int hash(int name, int index, uint32_t x);
};

/**
 * @brief 
 * 
 */
class MultiplyShiftHash : public BaseHash {
    /**
      * @brief A hash-map, containing the hashing methods, with strings as keys.
      * Acceptable keys are `"h1"`, `"h2"`, `"s1"` and `"s2"`
      */
    std::array<MatrixXui, 4> map;
  public:
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
    /**
     * @brief A vector of hash-maps, containing the hashing methods, with strings as keys.
     * Acceptable keys are `"h1"`, `"h2"`, `"s1"` and `"s2"`
     */
    std::array<std::vector<MatrixXui>, 4> map;
  public:
    TabulationHash(int p, int q, int r, int b, int d, std::mt19937_64 &rng);
    int hash(int name, int index, uint32_t x);
};

#endif
