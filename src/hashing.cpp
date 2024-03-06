#include "hashing.hpp"

#include <cmath>
#include <random>
#include <iostream>
#include <cstdint>

#include "utils.hpp"

Hashes<Eigen::MatrixXi> fully_random_constructor(int n, int b, int d, std::mt19937_64 &rng) {
    Eigen::MatrixXi h1(d, n), h2(d, n), s1(d, n), s2(d, n);

    std::uniform_int_distribution<int> bi(0, b - 1);

    for (int i = 0; i < d * n; i++) {
        h1.data()[i] = bi(rng);
        h2.data()[i] = bi(rng);
        s1.data()[i] = (rng() % 2 == 0) ? 1 : -1;
        s2.data()[i] = (rng() % 2 == 0) ? 1 : -1;
    }

    return {h1, h2, s1, s2};
}

Hashes<MatrixXui> multiply_shift_constructor(int d, std::mt19937_64 &rng) {
    MatrixXui h1(d, 2), h2(d, 2), s1(d, 2), s2(d, 2);

    std::uniform_int_distribution<uint64_t> uni(0, UINT64_MAX);

    for (int i = 0; i < 2 * d; i++) {
        h1.data()[i] = uni(rng);
        h2.data()[i] = uni(rng);
        s1.data()[i] = uni(rng);
        s2.data()[i] = uni(rng);
    }

    return {h1, h2, s1, s2};
}

Hashes<std::vector<MatrixXui>> tabulation_constructor(int p, int q, int r, int d, std::mt19937_64 &rng) {
    const int t = ceil(p / r);

    std::uniform_int_distribution<int> uni(0, pow(2, q));
    int size = pow(2, r);
    std::vector<MatrixXui> map;
    Hashes<std::vector<MatrixXui>> result;

    MatrixXui h1(t, size), h2(t, size), s1(t, size), s2(t, size);
    for (int k = 0; k < d; k++) {
        for (int i = 0; i < t * size; i++) {
            h1.data()[i] = uni(rng);
            h2.data()[i] = uni(rng);
            s1.data()[i] = uni(rng);
            s2.data()[i] = uni(rng);
        }

        result.h1.push_back(h1);
        result.h2.push_back(h2);
        result.s1.push_back(s1);
        result.s2.push_back(s2);
    }

    return result;
}
