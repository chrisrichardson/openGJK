
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

///
/// @brief Structure for a simplex.
///
using Simplex = struct Simplex
{
  /// Number of active vertices (1=point, 2=line, 3=triangle, 4=tetrahedron)
  int nvrtx;

  /// Vertex coordinates
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> vrtx;

  /// Vector
  Eigen::Vector3d vec;
};

double gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
           const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2,
           Simplex& s);