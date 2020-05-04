

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
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace
{
/// @brief Finds point of minimum norm of 1-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S1D(Simplex& s)
{
  assert(s.nvrtx == 2);
  const Eigen::Vector3d t = s.vrtx.row(0) - s.vrtx.row(1);

  const double tnorm2 = t.squaredNorm();
  if (tnorm2 > 0.0)
  {
    // Project origin onto the 1D simplex - line
    const double pt = s.vrtx.row(0).dot(t) / tnorm2;

    if (pt >= 0.0 and pt <= 1.0)
    {
      // The origin is between A and B
      s.vec = s.vrtx.row(0).transpose() - pt * t;
      return;
    }

    // The origin is beyond A, change point
    if (pt > 1.0)
      s.vrtx.row(0) = s.vrtx.row(1);
  }

  s.nvrtx = 1;
  s.vec = s.vrtx.row(0);
}

/// @brief Finds point of minimum norm of 2-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S2D(Simplex& s)
{
  assert(s.nvrtx == 3);
  const Eigen::Vector3d ac = s.vrtx.row(0) - s.vrtx.row(2);
  const Eigen::Vector3d bc = s.vrtx.row(0) - s.vrtx.row(1);
  const Eigen::Vector3d a = s.vrtx.row(2);
  const Eigen::Vector3d b = s.vrtx.row(1);

  // Find best axis for projection
  Eigen::Vector3d n = ac.cross(bc);
  const Eigen::Vector3d nu_fabs = n.cwiseAbs();
  int indexI;
  nu_fabs.maxCoeff(&indexI);
  const double nu_max = n[indexI];

  Eigen::Vector3d B;
  int FacetsTest[3] = {1, 1, 1};

  if (nu_max == 0.0)
  {
    if (ac.squaredNorm() == 0.0)
      FacetsTest[2] = 0;
    else
      FacetsTest[1] = 0;
  }
  else
  {
    n.normalize();
    const int indexJ0 = (indexI + 1) % 3;
    const int indexJ1 = (indexI + 2) % 3;
    const Eigen::Vector3d W1 = s.vrtx.col(indexJ0).head(3).reverse();
    const Eigen::Vector3d W2 = s.vrtx.col(indexJ1).head(3).reverse();
    const Eigen::Vector3d W3
        = (W1 * n[indexJ1] - W2 * n[indexJ0]) * n.dot(a + b) / 2.0;
    B = W1.cross(W2);
    B[0] += (W3[2] - W3[1]);
    B[1] += (W3[0] - W3[2]);
    B[2] += (W3[1] - W3[0]);

    // Test if sign of ABC is equal to the signs of the auxiliary simplices
    for (int i = 0; i < 3; ++i)
      FacetsTest[i] = (std::signbit(nu_max) == std::signbit(B[i]));
  }

  if ((FacetsTest[0] + FacetsTest[1] + FacetsTest[2]) == 3)
  {
    // The origin projection lays onto the triangle
    s.vec = s.vrtx.topRows(3).transpose() * B.reverse();
    s.vec /= nu_max;
    s.nvrtx = 3;
    return;
  }

  if (FacetsTest[1] == 0 and FacetsTest[2] == 0)
  {
    Simplex sTmp;
    sTmp.nvrtx = 2;
    sTmp.vrtx.row(0) = s.vrtx.row(1);
    sTmp.vrtx.row(1) = s.vrtx.row(2);
    S1D(sTmp);

    s.nvrtx = 2;
    s.vrtx.row(1) = s.vrtx.row(2);
    S1D(s);

    if (sTmp.vec.squaredNorm() < s.vec.squaredNorm())
      s = sTmp;
    return;
  }

  if (FacetsTest[2] == 0)
  {
    // The origin projection P faces the segment AB
    s.nvrtx = 2;
    s.vrtx.row(0) = s.vrtx.row(1);
    s.vrtx.row(1) = s.vrtx.row(2);
    S1D(s);
    return;
  }

  if (FacetsTest[1] == 0)
  {
    // The origin projection P faces the segment AC
    s.nvrtx = 2;
    s.vrtx.row(1) = s.vrtx.row(2);
    S1D(s);
    return;
  }

  // FacetsTest[0] == 0
  // The origin projection P faces the segment BC
  s.nvrtx = 2;
  S1D(s);
}

/// @brief Finds point of minimum norm of 3-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S3D(Simplex& s)
{
  assert(s.nvrtx == 4);

  Eigen::Vector4d B;
  const Eigen::Vector3d W1 = s.vrtx.row(0).cross(s.vrtx.row(1));
  const Eigen::Vector3d W2 = s.vrtx.row(2).cross(s.vrtx.row(3));
  B[0] = s.vrtx.row(2) * W1;
  B[1] = -s.vrtx.row(3) * W1;
  B[2] = s.vrtx.row(0) * W2;
  B[3] = -s.vrtx.row(1) * W2;

  double detM = B.sum();

  // Test if sign of ABCD is equal to the signs of the auxiliary simplices
  const double eps = 1e-15;
  int FacetsTest[4] = {1, 1, 1, 1};
  if (std::abs(detM) < eps)
  {
    Eigen::Vector4d Babs = B.cwiseAbs();
    if (Babs[2] < eps and Babs[3] < eps)
      FacetsTest[1] = 0; // A = B. Test only ACD
    else if (Babs[1] < eps and Babs[3] < eps)
      FacetsTest[2] = 0; // A = C. Test only ABD
    else if (Babs[1] < eps and Babs[2] < eps)
      FacetsTest[3] = 0; // A = D. Test only ABC
    else if (Babs[0] < eps and Babs[3] < eps)
      FacetsTest[1] = 0; // B = C. Test only ACD
    else if (Babs[0] < eps and Babs[2] < eps)
      FacetsTest[1] = 0; // B = D. Test only ACD
    else if (Babs[0] < eps and Babs[1] < eps)
      FacetsTest[2] = 0; // C = D. Test only ABD
    else
    {
      for (int i = 0; i < 4; i++)
        FacetsTest[i] = 0; // Any other case. Test ABC, ABD, ACD
    }
  }
  else
  {
    for (int i = 0; i < 4; ++i)
      FacetsTest[i] = (std::signbit(detM) == std::signbit(B[i]));
  }

  // Compare signed volumes and compute barycentric coordinates
  if (FacetsTest[0] + FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 4)
  {
    // All signs are equal, therefore the origin is inside the simplex
    s.vec.setZero();
    // s.vec = s.vrtx.transpose() * B.reverse();
    // s.vec /= detM;
    s.nvrtx = 4;
    return;
  }

  if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 3)
  {
    // The origin projection P faces the facet BCD
    s.nvrtx = 3;
    S2D(s);
    return;
  }

  // Either 1, 2 or 3 of ACD, ABD or ABC are closest.
  Simplex sBest;
  static const int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  double vmin = std::numeric_limits<double>::max();
  for (int i = 0; i < 3; ++i)
  {
    if (FacetsTest[i + 1] == 0)
    {
      Simplex sTmp;
      sTmp.nvrtx = 3;
      sTmp.vrtx.row(0) = s.vrtx.row(facets[i][0]);
      sTmp.vrtx.row(1) = s.vrtx.row(facets[i][1]);
      sTmp.vrtx.row(2) = s.vrtx.row(facets[i][2]);
      S2D(sTmp);
      const double vnorm = sTmp.vec.squaredNorm();
      if (vnorm < vmin)
      {
        vmin = vnorm;
        sBest = sTmp;
      }
    }
  }
  s = sBest;
}
} // namespace
//-----------------------------------------------------
double gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
           const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2)
{
  const int mk = 50; // Maximum number of iterations of the GJK algorithm
  const double eps2 = 1e-17; // Tolerance

  // Initialise
  Simplex s;
  s.nvrtx = 1;
  Eigen::Vector3d v = bd1.row(0) - bd2.row(0);
  s.vrtx.row(0) = v;

  // Begin GJK iteration
  int k;
  double vnorm2 = 0;
  for (k = 0; k < mk; ++k)
  {
    // Support function
    Eigen::VectorXd::Index i, j;
    (bd1 * -v).maxCoeff(&i);
    (bd2 * v).maxCoeff(&j);
    const Eigen::Vector3d w = bd1.row(i) - bd2.row(j);

    // 1st exit condition
    if (k > 0 && (vnorm2 - v.dot(w)) < eps2)
      break;

    // Simplex size should be less than 4, or should have exited already
    if (s.nvrtx == 4)
      throw std::runtime_error("OpenGJK error: simplex limit reached");

    // Add new vertex to simplex

    // Break if any existing points are the same as w
    int m;
    for (m = 0; m < s.nvrtx; ++m)
      if ((s.vrtx.row(m).array() == w.transpose().array()).all())
        break;
    if (m != s.nvrtx)
      break;

    s.vrtx.row(s.nvrtx) = w;
    ++s.nvrtx;

    if (k > 5)
      std::cout << "k = " << k << " s = \n[" << s.vrtx.topRows(s.nvrtx)
                << " vnorm2 = " << vnorm2 << ", " << (vnorm2 - v.dot(w))
                << "\n";

    // Invoke distance sub-algorithm
    switch (s.nvrtx)
    {
    case 4:
      S3D(s);
      break;
    case 3:
      S2D(s);
      break;
    case 2:
      S1D(s);
      break;
    }
    v = s.vec;
    vnorm2 = v.squaredNorm();
    // 2nd exit condition - intersecting or touching
    if (vnorm2 < eps2)
      break;
  }
  if (k == mk)
    throw std::runtime_error("OpenGJK error: max iteration limit reached");

#ifdef DEBUG
  std::cout << "OpenGJK iterations = " << k << "\n";
#endif

  // Compute and return distance
  return v.norm();
}
