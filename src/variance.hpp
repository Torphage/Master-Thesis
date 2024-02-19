#pragma once
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include <memory>
#include <vector>
#include <Eigen/Dense>

double variance(Eigen::VectorXd &vec);

Eigen::MatrixXd variance3d(std::vector<Eigen::MatrixXd> &mat);

#endif