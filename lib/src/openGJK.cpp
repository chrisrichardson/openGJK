

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
#include <iostream>

namespace
{
/// @brief Finds point of minimum norm of 1-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S1D(Simplex& s)
{
  assert(s.nvrtx == 2);
  const Eigen::Vector3d t = s.vrtx.row(0) - s.vrtx.row(1);

  // Project origin onto the 1D simplex - line
  const double pt = s.vrtx.row(0).dot(t) / t.squaredNorm();

  if (pt >= 0.0 and pt <= 1.0)
  {
    // The origin is between A and B
    s.vec = s.vrtx.row(0).transpose() - pt * t;
    return;
  }

  s.nvrtx = 1;
  // The origin is beyond A, change point
  if (pt > 1.0)
    s.vrtx.row(0) = s.vrtx.row(1);

  s.vec = s.vrtx.row(0);
}

/// @brief Finds point of minimum norm of 2-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S2D(Simplex& s)
{
  assert(s.nvrtx == 3);
  const Eigen::Vector3d c = s.vrtx.row(0);
  const Eigen::Vector3d b = s.vrtx.row(1);
  const Eigen::Vector3d a = s.vrtx.row(2);

  // Find best axis for projection
  Eigen::Vector3d n = a.cross(b) + b.cross(c) + c.cross(a);
  const Eigen::Vector3d nu_fabs = n.cwiseAbs();
  int indexI;
  nu_fabs.maxCoeff(&indexI);
  const double nu_max = n[indexI];
  const int indexJ0 = (indexI + 1) % 3;
  const int indexJ1 = (indexI + 2) % 3;
  n.normalize();
  const double dotNA = n.dot(a);

  Eigen::Vector3d B;
  Eigen::Matrix3d M;
  M.row(0) << s.vrtx(2, indexJ0), s.vrtx(2, indexJ1), 1.0;
  M.row(1) << s.vrtx(1, indexJ0), s.vrtx(1, indexJ1), 1.0;
  M.row(2) << dotNA * n[indexJ0], dotNA * n[indexJ1], 1.0;
  B[2] = M.determinant();
  M(1, 0) = s.vrtx(0, indexJ0);
  M(1, 1) = s.vrtx(0, indexJ1);
  B[1] = -M.determinant();
  M(0, 0) = s.vrtx(1, indexJ0);
  M(0, 1) = s.vrtx(1, indexJ1);
  B[0] = M.determinant();

  // Test if sign of ABC is equal to the signs of the auxiliary simplices
  int FacetsTest[3];
  for (int i = 0; i < 3; ++i)
    FacetsTest[i] = (std::signbit(nu_max) == std::signbit(B[i]));

  if ((FacetsTest[0] + FacetsTest[1] + FacetsTest[2]) == 3)
  {
    // The origin projections lays onto the triangle
    s.vec = s.vrtx.row(0) * B[2] + s.vrtx.row(1) * B[1] + s.vrtx.row(2) * B[0];
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

  Eigen::Matrix3d M = s.vrtx.bottomRows(3);
  Eigen::Vector4d B;
  B[3] = -M.determinant();
  M.row(0) = s.vrtx.row(0);
  B[2] = M.determinant();
  M.row(1) = s.vrtx.row(1);
  B[1] = -M.determinant();
  M.row(2) = s.vrtx.row(2);
  B[0] = M.determinant();
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
      FacetsTest[1] = 0; // B = D. Test only ABD
    else if (Babs[0] < eps and Babs[1] < eps)
      FacetsTest[2] = 0; // C = D. Test only ABC
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
    s.vec = s.vrtx.row(0) * B[3] + s.vrtx.row(1) * B[2] + s.vrtx.row(2) * B[1]
            + s.vrtx.row(3) * B[0];
    s.vec /= detM;
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
  Simplex sTmp, sBest;
  static const int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  double vmin = std::numeric_limits<double>::max();
  for (int i = 0; i < 3; ++i)
  {
    if (FacetsTest[i + 1] == 0)
    {
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
//-------------------------------------------------------
void support(
    Eigen::Vector3d& bs,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& body,
    const Eigen::Vector3d& v)
{
  Eigen::VectorXd::Index i;
  const double maxs = (body * v).maxCoeff(&i);
  if (maxs > bs.dot(v))
    bs = body.row(i);
}
} // namespace
//-----------------------------------------------------
double gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
           const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2)
{
  const int mk = 50; // Maximum number of iterations of the GJK algorithm
  const double eps_rel2 = 1e-25; // Tolerance on relative distance

  // Initialise
  Eigen::Vector3d bd1s = bd1.row(0);
  Eigen::Vector3d bd2s = bd2.row(0);
  Eigen::Vector3d v = bd1s - bd2s;
  Simplex s;
  s.nvrtx = 1;
  s.vrtx.row(0) = v;

  // Begin GJK iteration
  int k;
  for (k = 0; k < mk; ++k)
  {
    // Support function
    support(bd1s, bd1, -v);
    support(bd2s, bd2, v);

    const double vnorm2 = v.squaredNorm();
    const Eigen::Vector3d w = bd1s - bd2s;
    // 1st exit condition
    if (vnorm2 - v.dot(w) < eps_rel2)
      break;

    // 2nd exit condition - intersecting or touching
    if (vnorm2 < eps_rel2)
      break;

    // Simplex size should be less than 4, or should have exited already
    if (s.nvrtx == 4)
      throw std::runtime_error("OpenGJK error: simplex limit reached");
    // Add new vertex to simplex
    s.vrtx.row(s.nvrtx) = w;
    ++s.nvrtx;

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

    // Exit condition if no change in v
    if ((s.vec - v).squaredNorm() < 1e-30)
      break;
    v = s.vec;
  }
  if (k == mk)
    throw std::runtime_error("Max iteration limit reached");

#ifdef DEBUG
  std::cout << "OpenGJK iterations = " << k << "\n";
#endif

  // Compute and return distance
  return v.norm();
}
