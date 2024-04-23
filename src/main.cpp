#include "compressed_mul.hpp"
#include "function.hpp"
#include "hashing.hpp"
#include "utils.hpp"

#include <iostream>
#include <random>

int main() {
    int b = 50, d = 5, n = 10;

    unsigned int seed = 5;  // std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0, 1.0);

    MatrixRXd m1 = sparse_matrix_generator(n, 0.1, rng);
    MatrixRXd m2 = sparse_matrix_generator(n, 0.1, rng);

    // MatrixRXd m1 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});
    // MatrixRXd m2 = MatrixRXd::NullaryExpr(n,n,[&](){return uni(rng);});

    // FullyRandomHash<uint64_t> hash(n, b, d, seed);
    MultiplyShiftHash<uint32_t, uint16_t> hash(d, seed);
    // TabulationHash<uint32_t, uint32_t, 8> hash(d, seed);
    
    MatrixRXd compressed = function::compress_deluxe(m1, m2, n, b, d, hash);
    MatrixRXd result = function::decompress_par(compressed, n, b, d, hash);

    MatrixRXd real_product = m1.matrix() * m2.matrix();

    std::cout << "\n--------- Real result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(real_product) << std::endl;
    std::cout << real_product << std::endl;

    std::cout << "\n--------- Approximate result ---------" << std::endl;
    std::cout << "Sum of elements: " << sum_matrix(result) << std::endl;
    round_matrix(result, 12);
    std::cout << result << std::endl;

    return 0;
}

// #include <iostream>
// #include <Eigen/Dense>

// // Example method
// void myMethod(Eigen::MatrixXd &mat) {
//     mat(0,0) = 69;
//     // std::cout << "Inside myMethod: " << value << std::endl;
// }

// auto foo(int b, int d) {
//     Eigen::MatrixXd compressed = Eigen::MatrixXd::Zero(d, b);
//     compressed(0, 0) = 1;
//     compressed(1, 1) = 1;
//     compressed(0, 1) = 1;
//     compressed(1, 0) = 1;
//     return [&]() mutable -> Eigen::MatrixXd {
//         myMethod(compressed);
//         return compressed;
//     };
//     // std::cout << "Inside myMethod: " << value << std::endl;
// }

// int main() {
//     // Create a lambda that directly calls myMethod
//     // auto myLambda =  {
//     //     // Perform any additional logic here (if needed)
//     //     myMethod(value);
//     // };
//     auto a = foo(5, 4)();

//     // Invoke the lambda
//     // myLambda(42);
//     std::cout << a << std::endl;

//     return 0;
// }