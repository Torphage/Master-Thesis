#include <random>
#include "hashing.hpp"
#include <cmath>
#include <iostream>


BaseHash::BaseHash(int b, int d, std::mt19937_64 &rng) 
    : rng(rng)
    , b(b)
    , d(d)
    {}

FullyRandomHash::FullyRandomHash(int n, int b, int d, std::mt19937_64 &rng) 
: BaseHash(b, d, rng) {
    MatrixXui h1(d, n), h2(d, n), s1(d, n), s2(d, n);

    std::uniform_int_distribution<uint32_t> bi(0, b - 1);

    for (int i = 0; i < d * n; i++) {
        h1.data()[i] = bi(rng);
        h2.data()[i] = bi(rng);
        s1.data()[i] = (rng() % 2 == 0) ? 1 : -1;
        s2.data()[i] = (rng() % 2 == 0) ? 1 : -1;
    }
    
    map = {h1, h2, s1, s2};
}

int FullyRandomHash::hash(int name, int index, uint32_t x) {
    return map[name](index, x);
}

MultiplyShiftHash::MultiplyShiftHash(int b, int d, std::mt19937_64 &rng) 
: BaseHash(b, d, rng) {
    MatrixXui h1(d, 2), h2(d, 2), s1(d, 2), s2(d, 2);

    std::uniform_int_distribution<uint64_t> uni(0, pow(2, 64));

    for (int i = 0; i < 2 * d; i++) {
        h1.data()[i] = uni(rng);
        h2.data()[i] = uni(rng);
        s1.data()[i] = uni(rng);
        s2.data()[i] = uni(rng);
    }
    
    map = {h1, h2, s1, s2};
}

int MultiplyShiftHash::hash(int name, int index, uint32_t x) {
    uint64_t u = map[name](index, 0);
    uint64_t v = map[name](index, 1);

    uint64_t prod = u * x;
    uint64_t sum1 = prod + v;
    uint64_t intermediate = sum1 >> 32;
    
    uint64_t product;
    if (name == 0 || name == 1) {
        product = intermediate * b;
        return product >> 32;
    } else {
        product = intermediate * 2;
        return static_cast<int>(2 * (product >> 32)) - 1;
    }  
}


TabulationHash::TabulationHash(int p, int q, int r, int b, int d, std::mt19937_64 &rng) 
: BaseHash(b, d, rng), r(r) {
    t = ceil(p / r);

    std::uniform_int_distribution<int> uni(0, pow(2, q));
    int size = pow(2, r);

    MatrixXui h1(t, size), h2(t, size), s1(t, size), s2(t, size);
    for (int k = 0; k < d; k++) {
    
        for (int i = 0; i < t * size; i++) {
            h1.data()[i] = uni(rng);
            h2.data()[i] = uni(rng);
            s1.data()[i] = uni(rng);
            s2.data()[i] = uni(rng);
        }

        map[0].push_back(h1); // index 0
        map[1].push_back(h2); // index 1
        map[2].push_back(s1); // index 2
        map[3].push_back(s2); // index 3
    }
}

int TabulationHash::hash(int name, int index, uint32_t x) {    
    uint32_t res = 0;
    MatrixXui tab_matrix = map[name][index];
    int mask = (1 << r) - 1;

    for (int i = 0; i < t; i++) {
        uint32_t shift = x >> r * i;
        uint32_t masked = shift & mask;
        res ^= tab_matrix(i, masked);
    }

    if (name == 0 || name == 1) {
        return res % b;
    } else {
        return 2 * static_cast<int>(res % 2) - 1;
    }
}

