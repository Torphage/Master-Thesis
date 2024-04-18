// #include <benchmark/benchmark.h>
// #include <omp.h>

// // static void BM_func(benchmark::State& state) {
// //     int n = state.range(0);

// //     std::vector<std::vector<double>> m1(n, std::vector<double>(n, 0));
// //     std::vector<std::vector<double>> m2(n, std::vector<double>(n, 0));
// //     std::vector<std::vector<double>> c(n, std::vector<double>(n, 0));

// //     for (int i = 0; i < n; i++) {
// //         for (int j = 0; j < n; j++) {
// //             m1[i][j] = drand48();
// //         }        
// //     }
// //     for (int i = 0; i < n; i++) {
// //         for (int j = 0; j < n; j++) {
// //             m2[i][j] = drand48();
// //         }        
// //     }

// //     for (auto _ : state) {
// //         for (int i = 0; i < n; i++) {
// //             for (int j = 0; j < n; j++) {
// //                 for (int k = 0; k < n; k++) {
// //                     c[i][j] += m1[i][k] * m2[k][j];
// //                 }
// //             }
// //         }
// //     }
// // }

// // static void BM_bompressed(benchmark::State& state) {
// // int n = state.range(0);

// //     std::vector<std::vector<double>> m1(n, std::vector<double>(n, 0));
// //     std::vector<std::vector<double>> m2(n, std::vector<double>(n, 0));
// //     std::vector<std::vector<double>> c(n, std::vector<double>(n, 0));

// //     for (int i = 0; i < n; i++) {
// //         for (int j = 0; j < n; j++) {
// //             m1[i][j] = drand48();
// //         }        
// //     }
// //     for (int i = 0; i < n; i++) {
// //         for (int j = 0; j < n; j++) {
// //             m2[i][j] = drand48();
// //         }        
// //     }

// //     for (auto _ : state) {
// // #pragma omp parallel for 
// //         for (int i = 0; i < n; i++) {
// //             for (int j = 0; j < n; j++) {
// //                 for (int k = 0; k < n; k++) {
// //                     c[i][j] += m1[i][k] * m2[k][j];
// //                 }
// //             }
// //         }
// //     }
// // }

// // static void func(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2, int n) {
// //     for (int i = 0; i < n; i++) {
// //         for (int j = 0; j < n; j++) {
// //             for (int k = 0; k < n; k++) {
// //                 c[i][j] += m1[i][k] * m2[k][j];
// //             }
// //         }
// //     }
// // }

// // static void func_par(MatrixRXd &c, MatrixRXd &m1, MatrixRXd &m2, int n) {
// // #pragma omp parallel for 
// //     for (int i = 0; i < n; i++) {
// //         for (int j = 0; j < n; j++) {
// //             for (int k = 0; k < n; k++) {
// //                 c[i][j] += m1[i][k] * m2[k][j];
// //             }
// //         }
// //     }
// // }

// // BENCHMARK(BM_func)->Iterations(7)->Arg(700)->Unit(benchmark::kMillisecond);
// // BENCHMARK(BM_bompressed)->Iterations(7)->Arg(700)->Unit(benchmark::kMillisecond);

// /**
//  * @brief 
//  * 
//  * @param argc 
//  * @param argv The first argument must be the number of threads to use
//  * @return int 
//  */
// int main(int argc, char** argv) {
//     omp_set_num_threads(2);

//     benchmark::Initialize(&argc, argv);

    
//     benchmark::RunSpecifiedBenchmarks();
//     benchmark::Shutdown();
//     return 0;   
// }

int main() {return 0;}