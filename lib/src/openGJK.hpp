
#pragma once

#include <Eigen/Dense>

///
/// @brief Structure for a simplex.
///
using Simplex = struct Simplex
{
  /// Vertex coordinates
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> vrtx;

  /// Number of active vertices (1=point, 2=line, 3=triangle, 4=tetrahedron)
  int nvrtx;
};

/// Calculate the distance between two convex bodies bd1 and bd2
/// @param[in] bd1 Body 1 list of vertices
/// @param[in] bd2 Body 2 list of vertices
/// @return distance between bodies
Eigen::Vector3d
gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2);