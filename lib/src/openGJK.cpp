

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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

/**
 * @brief Structure for a simplex.
 */
class Simplex
{
public:
  /// Return a vector based on current coordinates and lambdas
  Eigen::Vector3d vec() const
  {
    Eigen::Vector3d vv;
    vv.setZero();
    for (int i = 0; i < nvrtx; ++i)
      vv += lambdas[i] * vrtx.row(i);
    return vv;
  }

  /// Number of active vertices (1=point, 2=line, 3=triangle, 4=tetrahedron)
  int nvrtx;
  /// Vertex coordinates
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> vrtx;
  /// Vertex IDs
  Eigen::Vector4i wids;
  /// Barycentric coordinates
  Eigen::Vector4d lambdas;
};

/// @brief Finds point of minimum norm of 1-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S1D(Simplex& s)
{
  Eigen::Vector3d t = s.vrtx.row(0) - s.vrtx.row(1);

  // Project origin onto the 1D simplex - line
  double pt = s.vrtx.row(0).dot(t) / t.squaredNorm();

  if (pt >= 0.0 and pt <= 1.0)
  {
    // The origin is between A and B
    s.lambdas[0] = 1.0 - pt;
    s.lambdas[1] = pt;
    s.wids[0] = 0;
    s.wids[1] = 1;
    s.nvrtx = 2;
    return;
  }

  if (pt > 1.0)
  {
    // The origin is beyond A
    s.lambdas[0] = 1;
    s.wids[0] = 0;
    s.nvrtx = 1;
    s.vrtx.row(0) = s.vrtx.row(1);
    return;
  }

  // pt < 0.0
  // The origin is behind B
  s.lambdas[0] = 1;
  s.wids[0] = 1;
  s.nvrtx = 1;
}

/// @brief Finds point of minimum norm of 2-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S2D(Simplex& s)
{
  const Eigen::Vector3d c = s.vrtx.row(0);
  const Eigen::Vector3d b = s.vrtx.row(1);
  const Eigen::Vector3d a = s.vrtx.row(2);

  // Find best axis for projection
  Eigen::Vector3d n = a.cross(b) + b.cross(c) + c.cross(a);
  Eigen::Vector3d nu_fabs = n.cwiseAbs();
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
  M.row(1) << s.vrtx(0, indexJ0), s.vrtx(0, indexJ1), 1.0;
  B[1] = -M.determinant();
  M.row(0) << s.vrtx(1, indexJ0), s.vrtx(1, indexJ1), 1.0;
  B[0] = M.determinant();

  // Test if sign of ABC is equal to the signs of the auxiliary simplices
  int FacetsTest[3];
  for (int i = 0; i < 3; ++i)
    FacetsTest[i] = (std::signbit(nu_max) == std::signbit(B[i]));

  if ((FacetsTest[0] + FacetsTest[1] + FacetsTest[2]) == 3)
  {
    // The origin projections lays onto the triangle
    const double inv_detM = 1 / nu_max;
    s.lambdas[0] = B[2] * inv_detM;
    s.lambdas[1] = B[1] * inv_detM;
    s.lambdas[2] = 1.0 - s.lambdas[0] - s.lambdas[1];
    s.wids[0] = 0;
    s.wids[1] = 1;
    s.wids[2] = 2;
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

    if (s.vec().squaredNorm() < sTmp.vec().squaredNorm())
    {
      // Keep simplex. Update sID.
      s.wids[1] = 2;
    }
    else
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
    s.wids[1] = 2;
    return;
  }

  // FacetsTest[0] == 0
  // The origin projection P faces the segment BC
  s.nvrtx = 2;
  S1D(s);
}
//-------------------------------------------------------
/// @brief Finds point of minimum norm of 3-simplex. Robust, but slower,
/// version of algorithm presented in paper.
void S3D(Simplex& s)
{
  int FacetsTest[4] = {1, 1, 1, 1};
  static const int TrianglesToTest[9] = {3, 3, 3, 1, 2, 2, 0, 0, 1};
  int firstaux = 0;
  int secondaux = 0;
  double sqdist_tmp = 0;

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
  double eps = 1e-13;
  if (std::abs(detM) < eps)
  {
    Eigen::Vector4d Babs = B.cwiseAbs();
    if (Babs[2] < eps and Babs[3] < eps)
      FacetsTest[1] = 0; /* A = B. Test only ACD */
    else if (Babs[1] < eps and Babs[3] < eps)
      FacetsTest[2] = 0; /* A = C. Test only ABD */
    else if (Babs[1] < eps and Babs[2] < eps)
      FacetsTest[3] = 0; /* A = D. Test only ABC */
    else if (Babs[0] < eps and Babs[3] < eps)
      FacetsTest[1] = 0; /* B = C. Test only ACD */
    else if (Babs[0] < eps and Babs[2] < eps)
      FacetsTest[1] = 0; /* B = D. Test only ABD */
    else if (Babs[0] < eps and Babs[1] < eps)
      FacetsTest[2] = 0; /* C = D. Test only ABC */
    else
    {
      for (int i = 0; i < 4; i++)
        FacetsTest[i] = 0; /* Any other case. Test ABC, ABD, ACD */
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
    const double inv_detM = 1 / detM;
    s.lambdas[3] = B[0] * inv_detM;
    s.lambdas[2] = B[1] * inv_detM;
    s.lambdas[1] = B[2] * inv_detM;
    s.lambdas[0] = 1 - s.lambdas[1] - s.lambdas[2] - s.lambdas[3];
    s.wids[0] = 0;
    s.wids[1] = 1;
    s.wids[2] = 2;
    s.wids[3] = 3;
    s.nvrtx = 4;
    return;
  }

  if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 0)
  {
    // There are three facets facing the origin
    Simplex sTmp;
    Eigen::Vector4d tmplamda;
    sqdist_tmp = std::numeric_limits<double>::max();
    int nclosestS = 0;
    int sID[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
      sTmp.nvrtx = 3;
      // Assign coordinates to simplex
      for (int k = 0; k < sTmp.nvrtx; ++k)
      {
        const int vtxid = TrianglesToTest[i + (k * 3)];
        sTmp.vrtx.row(2 - k) = s.vrtx.row(vtxid);
      }
      S2D(sTmp);
      double vtmp_norm2 = sTmp.vec().squaredNorm();

      // ... compute aux squared distance
      if (vtmp_norm2 < sqdist_tmp)
      {
        sqdist_tmp = vtmp_norm2;
        nclosestS = sTmp.nvrtx;
        tmplamda = sTmp.lambdas;
        for (int l = 0; l < nclosestS; ++l)
          sID[l] = TrianglesToTest[i + (sTmp.wids[l] * 3)];
      }
    }

    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> tmpscoo1 = s.vrtx;

    /* Store closest simplex */
    s.nvrtx = nclosestS;
    s.lambdas = tmplamda;
    for (int i = 0; i < s.nvrtx; ++i)
    {
      s.vrtx.row(nclosestS - 1 - i) = tmpscoo1.row(sID[i]);
      s.wids[nclosestS - 1 - i] = sID[i];
    }
    return;
  }

  if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 1)
  {
    // There are two facets facing the origin, need to find the closest.
    Simplex sTmp;
    sTmp.nvrtx = 3;

    if (FacetsTest[1] == 0)
    {
      /* ... Test facet ACD  */
      sTmp.vrtx.row(0) = s.vrtx.row(0);
      sTmp.vrtx.row(1) = s.vrtx.row(1);
      sTmp.vrtx.row(2) = s.vrtx.row(3);
      /* ... and call S2D  */
      S2D(sTmp);
      /* ... compute aux squared distance */
      sqdist_tmp = sTmp.vec().squaredNorm();
      firstaux = 0;
    }
    if (FacetsTest[2] == 0)
    {
      /* ... Test facet ABD  */
      if (FacetsTest[1] == 1)
      {
        sTmp.vrtx.row(0) = s.vrtx.row(0);
        sTmp.vrtx.row(1) = s.vrtx.row(2);
        sTmp.vrtx.row(2) = s.vrtx.row(3);
        /* ... and call S2D  */
        S2D(sTmp);
        /* ... compute aux squared distance */
        sqdist_tmp = sTmp.vec().squaredNorm();
        firstaux = 1;
      }
      else
      {
        s.nvrtx = 3;
        s.vrtx.row(1) = s.vrtx.row(2);
        s.vrtx.row(2) = s.vrtx.row(3);
        /* ... and call S2D itself */
        S2D(s);
        secondaux = 1;
      }
    }

    if (FacetsTest[3] == 0)
    {
      /* ... Test facet ABC  */
      s.nvrtx = 3;
      s.vrtx.row(0) = s.vrtx.row(1);
      s.vrtx.row(1) = s.vrtx.row(2);
      s.vrtx.row(2) = s.vrtx.row(3);
      /* ... and call S2D  */
      S2D(s);
      secondaux = 2;
    }

    // Compare outcomes
    if (s.vec().squaredNorm() < sqdist_tmp)
    {
      /* Keep simplex. Need to update sID only*/
      for (int i = 0; i < s.nvrtx; ++i)
      {
        /* Assume that vertex a is always included in sID. */
        s.wids[s.nvrtx - 1 - i] = TrianglesToTest[secondaux + (s.wids[i] * 3)];
      }
    }
    else
    {
      s.nvrtx = sTmp.nvrtx;
      s.vrtx = sTmp.vrtx;
      s.lambdas = sTmp.lambdas;
      for (int i = 0; i < s.nvrtx; ++i)
      {
        s.wids[sTmp.nvrtx - 1 - i]
            = TrianglesToTest[firstaux + (sTmp.wids[i] * 3)];
      }
    }
    return;
  }

  if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 2)
  {
    // Only one facet is facing the origin
    if (FacetsTest[1] == 0)
    {
      // The origin projection P faces the facet ACD
      s.nvrtx = 3;
      s.vrtx.row(1) = s.vrtx.row(1);
      s.vrtx.row(2) = s.vrtx.row(3);
      S2D(s);
    }
    else if (FacetsTest[2] == 0)
    {
      // The origin projection P faces the facet ABD
      s.nvrtx = 3;
      s.vrtx.row(1) = s.vrtx.row(2);
      s.vrtx.row(2) = s.vrtx.row(3);
      S2D(s);
      // Keep simplex. Need to update sID only
      // Assume that vertex a is always included in sID.
      for (int i = 2; i < s.nvrtx; ++i)
        ++s.wids[i];
    }
    else if (FacetsTest[3] == 0)
    {
      // The origin projection P faces the facet ABC
      s.nvrtx = 3;
      s.vrtx.row(0) = s.vrtx.row(1);
      s.vrtx.row(1) = s.vrtx.row(2);
      s.vrtx.row(2) = s.vrtx.row(3);
      S2D(s);
    }
    return;
  }

  // The origin projection P faces the facet BCD
  s.nvrtx = 3;
  S2D(s);
  // Keep simplex. Need to update sID only
  // Assume that vertex a is always included in sID.
  for (int i = 0; i < s.nvrtx; ++i)
    ++s.wids[i];
}
//-------------------------------------------------------
void support(
    Eigen::Vector3d& bs,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& body,
    const Eigen::Vector3d& v)
{
  Eigen::VectorXd::Index i;
  double maxs = (body * v).maxCoeff(&i);
  if (maxs > bs.dot(v))
    bs = body.row(i);
}
//-----------------------------------------------------
void subalgorithm(Simplex& s)
{
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
}
//-------------------------------------------------------------
double gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
           const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2,
           Simplex& s)
{
  int k = 0;   /**< Iteration counter            */
  int mk = 50; /**< Maximum number of iterations of the GJK algorithm */
  double eps_rel = 1e-5;  /**< Tolerance on relative distance */
  double eps_tot = 1e-15; /**< Tolerance on absolute distance */

  double eps_rel2 = eps_rel * eps_rel;

  /* Initialise  */
  s.nvrtx = 1;
  Eigen::Vector3d bd1s = bd1.row(0);
  Eigen::Vector3d bd2s = bd2.row(0);
  Eigen::Vector3d v = bd1s - bd2s;
  s.vrtx.row(0) = v;

  /* Begin GJK iteration */
  do
  {
    k++;

    /* Support function */
    support(bd1s, bd1, -v);
    support(bd2s, bd2, v);

    const double vnorm2 = v.squaredNorm();
    Eigen::Vector3d w = bd1s - bd2s;
    /* 1st exit condition */
    double exeedtol_rel = (vnorm2 - v.dot(w)) <= eps_rel2 * vnorm2;
    if (exeedtol_rel)
      break;

    /* 2nd exit condition */
    if (vnorm2 < eps_rel2)
      break;

    /* Add new vertex to simplex */
    s.vrtx.row(s.nvrtx) = w;
    ++s.nvrtx;

    /* Invoke distance sub-algorithm */
    subalgorithm(s);
    v = s.vec();

    /* 3rd exit condition */
    double norm2Wmax = 0;
    for (int i = 0; i < s.nvrtx; ++i)
      norm2Wmax = std::max(norm2Wmax, s.vrtx.row(i).squaredNorm());
    if (v.squaredNorm() <= (eps_tot * eps_tot * norm2Wmax))
      break;

    /* 4th and 5th exit conditions */
  } while (s.nvrtx != 4 and k != mk);

  /* Compute and return distance */
  return v.norm();
}

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(opengjk, m)
{
  m.def("gjk",
        [](Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> arr1,
           Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> arr2)
            -> double {
          Simplex s;
          return gjk(arr1, arr2, s);
        });
}
