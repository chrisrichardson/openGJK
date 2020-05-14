

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
 *                                   #####        # #    #                *
 *       ####  #####  ###### #    # #     #       # #   #                 *
 *      #    # #    # #      ##   # #             # #  #                  *
 *      #    # #    # #####  # #  # #  ####       # ###                   *
 *      #    # #####  #      #  # # #     # #     # #  #                  *
 *      #    # #      #      #   ## #     # #     # #   #                 *
 *       ####  #      ###### #    #  #####   #####  #    #                *
 *                                                                        *
 *  This file is part of openGJK.                                         *
 *                                                                        *
 *       openGJK: open-source Gilbert-Johnson-Keerthi algorithm           *
 *            Copyright (C) Mattia Montanari 2018 - 2019                  *
 *              http://iel.eng.ox.ac.uk/?page_id=504                      *
 *                                                                        *
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
 *                                                                        *
 * This file implements the GJK algorithm and the Signed Volumes method as*
 * presented in:                                                          *
 *   M. Montanari, N. Petrinic, E. Barbieri, "Improving the GJK Algorithm *
 *   for Faster and More Reliable Distance Queries Between Convex Objects"*
 *   ACM Transactions on Graphics, vol. 36, no. 3, Jun. 2017.             *
 *                                                                        *
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

#include "openGJK.hpp"
#include <Eigen/Geometry>
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace
{
/// @brief Finds point of minimum norm of 1-simplex. Robust, but slower,
/// version of algorithm presented in paper.
std::pair<std::array<int, 4>, Eigen::Vector3d>
S1D(const Eigen::Matrix<double, 4, 3, Eigen::RowMajor>& s, int r0, int r1)
{
  const Eigen::RowVector3d t = s.row(r0) - s.row(r1);

  const double tnorm2 = t.squaredNorm();
  if (tnorm2 == 0.0)
    throw std::runtime_error("t=0 in S1D");

  // Project origin onto the 1D simplex - line
  const double pt = s.row(r0).dot(t) / tnorm2;

  if (pt >= 0.0 and pt <= 1.0)
  {
    // The origin is between A and B
    Eigen::Vector3d v = s.row(r0) - pt * t;
    return {{r0, r1, -1, -1}, v};
  }

  // The origin is beyond A, change point
  if (pt > 1.0)
    return {{r1, -1, -1, -1}, s.row(r1)};
  else
    return {{r0, -1, -1, -1}, s.row(r0)};
}

/// @brief Finds point of minimum norm of 2-simplex. Robust, but slower,
/// version of algorithm presented in paper.
std::pair<std::array<int, 4>, Eigen::Vector3d>
S2D(const Eigen::Matrix<double, 4, 3, Eigen::RowMajor>& s, int r0, int r1,
    int r2)
{
  Eigen::Matrix3d M;
  M.row(0) = s.row(r0);
  M.row(1) = s.row(r1);
  M.row(2) = s.row(r2);
  const Eigen::Vector3d ac = M.row(0) - M.row(2);
  const Eigen::Vector3d bc = M.row(0) - M.row(1);

  // Find best axis for projection
  Eigen::Vector3d n = ac.cross(bc);
  if (n.squaredNorm() == 0.0)
    throw std::runtime_error("Zero normal in S2D");
  const Eigen::Vector3d nu_fabs = n.cwiseAbs();
  int indexI;
  nu_fabs.maxCoeff(&indexI);
  const double nu_max = n[indexI];

  // Barycentre of triangle
  Eigen::Vector3d p = M.colwise().sum() / 3.0;
  // Renormalise nvrtx in plane of ABC
  n *= n.dot(p) / n.squaredNorm();

  const int indexJ0 = (indexI + 1) % 3;
  const int indexJ1 = (indexI + 2) % 3;
  const Eigen::Vector3d W1 = M.col(indexJ0).reverse();
  const Eigen::Vector3d W2 = M.col(indexJ1).reverse();
  const Eigen::Vector3d W3 = (W1 * n[indexJ1] - W2 * n[indexJ0]);
  Eigen::Vector3d B = W1.cross(W2);
  B[0] += (W3[2] - W3[1]);
  B[1] += (W3[0] - W3[2]);
  B[2] += (W3[1] - W3[0]);

  // Test if sign of ABC is equal to the signs of the auxiliary simplices
  std::array<bool, 3> FacetsTest;
  for (int i = 0; i < 3; ++i)
    FacetsTest[i] = (std::signbit(nu_max) == std::signbit(B[i]));

  if (FacetsTest[0] and FacetsTest[1] and FacetsTest[2])
  {
    // The origin projection lays onto the triangle
    return {{r0, r1, r2, -1}, n};
  }

  std::array<int, 4> arrmin = {-1, -1, -1, -1};
  Eigen::Vector3d vmin = {0, 0, 0};

  // Test if the origin projection P faces the segment BC
  if (FacetsTest[1] and FacetsTest[2])
    std::tie(arrmin, vmin) = S1D(s, r0, r1);

  double qmin = std::numeric_limits<double>::max();
  if (FacetsTest[1] == false)
  {
    std::tie(arrmin, vmin) = S1D(s, r0, r2);
    qmin = vmin.squaredNorm();
  }
  if (FacetsTest[2] == false)
  {
    auto [arr, v] = S1D(s, r1, r2);
    if (v.squaredNorm() < qmin)
    {
      arrmin = arr;
      vmin = v;
    }
  }

  return {arrmin, vmin};
}

/// @brief Finds point of minimum norm of 3-simplex. Robust, but slower,
/// version of algorithm presented in paper.
std::pair<std::array<int, 4>, Eigen::Vector3d>
S3D(const Eigen::Matrix<double, 4, 3, Eigen::RowMajor>& s)
{
  Eigen::Vector4d B;
  const Eigen::Vector3d W1 = s.row(0).cross(s.row(1));
  const Eigen::Vector3d W2 = s.row(2).cross(s.row(3));
  B[0] = s.row(2) * W1;
  B[1] = -s.row(3) * W1;
  B[2] = s.row(0) * W2;
  B[3] = -s.row(1) * W2;
  double detM = B.sum();

  // Test if sign of ABCD is equal to the signs of the auxiliary simplices
  std::array<bool, 4> FacetsTest;
  for (int i = 0; i < 4; ++i)
    FacetsTest[i] = (std::signbit(detM) == std::signbit(B[i]));

  if (FacetsTest[0] and FacetsTest[1] and FacetsTest[2] and FacetsTest[3])
  {
    // All signs are equal, therefore the origin is inside the simplex
    return {{0, 1, 2, 3}, Eigen::Vector3d::Zero()};
  }
  else if (FacetsTest[1] and FacetsTest[2] and FacetsTest[3])
  {
    // The origin projection P faces the facet BCD
    return S2D(s, 0, 1, 2);
  }

  // Test ACD, ABD and/or ABC.
  std::array<int, 4> arrmin = {-1, -1, -1, -1};
  Eigen::Vector3d vmin = {0, 0, 0};
  static const int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  double qmin = std::numeric_limits<double>::max();
  for (int i = 0; i < 3; ++i)
  {
    if (FacetsTest[i + 1] == false)
    {
      auto [arr, v] = S2D(s, facets[i][0], facets[i][1], facets[i][2]);

      const double q = v.squaredNorm();
      if (q < qmin)
      {
        qmin = q;
        vmin = v;
        arrmin = arr;
      }
    }
  }
  return {arrmin, vmin};
}
//-------------------------------------------------------------------------------
Eigen::Vector3d
support(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd,
        const Eigen::Vector3d& v)
{
  int i = 0;
  double qmax = bd.row(0) * v;
  for (int m = 1; m < bd.rows(); ++m)
  {
    const double q = bd.row(m) * v;
    if (q > qmax)
    {
      qmax = q;
      i = m;
    }
  }
  return bd.row(i);
}

} // namespace
//-----------------------------------------------------
Eigen::Vector3d
gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2)
{
  const int maxk = 50; // Maximum number of iterations of the GJK algorithm

  // Tolerances
  const double eps_tot = 1e-12;
  const double eps_rel = 1e-12;

  // Initialise
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> s;
  int nvrtx = 1;
  std::array<int, 4> pt_index = {0, -1, -1, -1};
  Eigen::Vector3d v = bd1.row(0) - bd2.row(0);
  s.row(0) = v;

  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {
    // Support function
    const Eigen::Vector3d w = support(bd1, -v) - support(bd2, v);

    // Break if any existing points are the same as w
    int m;
    for (m = 0; m < nvrtx; ++m)
      if (s(m, 0) == w[0] and s(m, 1) == w[1] and s(m, 2) == w[2])
        break;
    if (m != nvrtx)
      break;

    // 1st exit condition (v-w).v = 0
    const double vnorm2 = v.squaredNorm();
    const double vw = vnorm2 - v.dot(w);
    if (vw < (eps_rel * vnorm2) or vw < eps_tot)
      break;

    // Add new vertex to simplex
    s.row(nvrtx) = w;
    ++nvrtx;

    // Invoke distance sub-algorithm
    if (nvrtx == 4)
      std::tie(pt_index, v) = S3D(s);
    else if (nvrtx == 3)
      std::tie(pt_index, v) = S2D(s, 0, 1, 2);
    else
      std::tie(pt_index, v) = S1D(s, 0, 1);

    // Rebuild s
    for (nvrtx = 0; nvrtx < 3; ++nvrtx)
    {
      if (pt_index[nvrtx] == -1)
        break;
      s.row(nvrtx) = s.row(pt_index[nvrtx]);
    }

    // 2nd exit condition - intersecting or touching
    if (v.squaredNorm() < eps_rel * eps_rel)
      break;
  }
  if (k == maxk)
    throw std::runtime_error("OpenGJK error: max iteration limit reached");

  // Compute and return distance
  return v;
}
