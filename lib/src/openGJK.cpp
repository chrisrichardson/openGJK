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
 *  openGJK is free software: you can redistribute it and/or modify       *
 *   it under the terms of the GNU General Public License as published by *
 *   the Free Software Foundation, either version 3 of the License, or    *
 *   any later version.                                                   *
 *                                                                        *
 *   openGJK is distributed in the hope that it will be useful,           *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See The        *
 *   GNU General Public License for more details.                         *
 *                                                                        *
 *  You should have received a copy of the GNU General Public License     *
 *   along with Foobar.  If not, see <https://www.gnu.org/licenses/>.     *
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

#define SAMESIGN(a, b) ((a > 0) == (b > 0))

/**
 * @brief Structure for a simplex.
 */
class Simplex
{
public:
  int nvrtx;
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> vrtx;
  Eigen::Vector4i wids;
  Eigen::Vector4d lambdas;
  // Eigen::Matrix<double, 4, 3, Eigen::RowMajor> p;
  // Eigen::Matrix<double, 4, 3, Eigen::RowMajor>  q;
};

/**
 * @file openGJK.c
 * @author Mattia Montanari
 * @date April 2018
 * @brief File containing entire implementation of the GJK algorithm.
 *
 */

/**
 * @brief Finds point of minimum norm of 1-simplex. Robust, but slower,
 * version of algorithm presented in paper.
 */
Eigen::Vector3d S1D(Simplex& s)
{
  Eigen::Vector3d b = s.vrtx.row(0);
  Eigen::Vector3d a = s.vrtx.row(1);
  Eigen::Vector3d t = b - a;

  /* Project origin onto the 1D simplex - line */
  double pt = b.dot(t) / t.squaredNorm();

  if (pt >= 0.0 and pt <= 1.0)
  {
    /* The origin is between A and B */
    s.lambdas[0] = 1.0 - pt;
    s.lambdas[1] = pt;
    s.wids[0] = 0;
    s.wids[1] = 1;
    s.nvrtx = 2;
  }
  else if (pt > 1.0)
  {
    /* The origin is beyond A */
    s.lambdas[0] = 1;
    s.wids[0] = 0;
    s.nvrtx = 1;
    s.vrtx.row(0) = s.vrtx.row(1);
  }
  else
  {
    /* The origin is behind B */
    s.lambdas[0] = 1;
    s.wids[0] = 1;
    s.nvrtx = 1;
  }

  Eigen::Vector3d vv;
  vv.setZero();
  for (int i = 0; i < s.nvrtx; ++i)
    vv += s.lambdas[i] * s.vrtx.row(i);
  return vv;
}

/**
 * @brief Finds point of minimum norm of 2-simplex. Robust, but slower,
 * version of algorithm presented in paper.
 */
void S2D(Simplex& s, Eigen::Vector3d& vv)
{
  Eigen::Vector3d v, vtmp;
  Eigen::Vector3d a, b, c, s21, s31;
  c = s.vrtx.row(0);
  b = s.vrtx.row(1);
  a = s.vrtx.row(2);
  s21 = b - a;
  s31 = c - a;

  /* Find best axis for projection */
  Eigen::Vector3d nu_test = a.cross(b) + b.cross(c) + c.cross(a);
  Eigen::Vector3d nu_fabs = nu_test.cwiseAbs();
  int indexI;
  nu_fabs.maxCoeff(&indexI);
  double nu_max = nu_test[indexI];
  int indexJ[2] = {(indexI + 1) % 3, (indexI + 2) % 3};

  // int k = 1;
  // int l = 2;
  // Eigen::Vector3d n;
  // for (int i = 0; i < 3; ++i)
  // {
  //   n[i] = s21[k] * s31[l] - s21[l] * s31[k];
  //   k = l;
  //   l = i;
  // }
  Eigen::Vector3d n = s21.cross(s31);
  n.normalize();

  double dotNA = n.dot(a);

  double pp[2] = {dotNA * n[indexJ[0]], dotNA * n[indexJ[1]]};

  /* Compute signed determinants */
#ifndef ADAPTIVEFP
  double ss[3][3 - 1];
  ss[0][0] = a[indexJ[0]];
  ss[0][1] = a[indexJ[1]];
  ss[1][0] = b[indexJ[0]];
  ss[1][1] = b[indexJ[1]];
  ss[2][0] = c[indexJ[0]];
  ss[2][1] = c[indexJ[1]];

  int k = 1;
  int l = 2;
  Eigen::Vector3d B;
  for (int i = 0; i < 3; ++i)
  {
    B[i] = pp[0] * ss[k][1] + pp[1] * ss[l][0] + ss[k][0] * ss[l][1]
           - pp[0] * ss[l][1] - pp[1] * ss[k][0] - ss[l][0] * ss[k][1];
    k = l;
    l = i;
  }
#else
  double sa[2], sb[2], sc[2];
  sa[0] = a[indexJ[0]];
  sa[1] = a[indexJ[1]];
  sb[0] = b[indexJ[0]];
  sb[1] = b[indexJ[1]];
  sc[0] = c[indexJ[0]];
  sc[1] = c[indexJ[1]];

  B[0] = orient2d(sa, pp, sc);
  B[1] = orient2d(sa, pp, sc);
  B[2] = orient2d(sa, sb, pp);
#endif

  /* Test if sign of ABC is equal to the signes of the auxiliary simplices */
  int FacetsTest[3];
  for (int i = 0; i < 3; ++i)
    FacetsTest[i] = SAMESIGN(nu_max, B[i]);

  // The nan check was not included originally and will be removed in v2.0
  if (FacetsTest[1] + FacetsTest[2] == 0 or std::isnan(n[0]))
  {
    Simplex sTmp;

    sTmp.nvrtx = 2;
    s.nvrtx = 2;

    sTmp.vrtx.row(0) = s.vrtx.row(1);
    sTmp.vrtx.row(1) = s.vrtx.row(2);
    // s.vrtx.row(0) = s.vrtx.row(0);
    s.vrtx.row(1) = s.vrtx.row(2);

    vtmp = S1D(sTmp);
    v = S1D(s);

    if (v.squaredNorm() < vtmp.squaredNorm())
    {
      /* Keep simplex. Need to update sID only*/
      for (int i = 1; i < s.nvrtx; ++i)
        ++s.wids[i];
    }
    else
    {
      s.nvrtx = sTmp.nvrtx;
      s.vrtx = sTmp.vrtx;
      s.lambdas = sTmp.lambdas;

      for (int i = 0; i < s.nvrtx; ++i)
      {
        /* No need to convert sID here since sTmp deal with the vertices A
         * and B. ;*/
        s.wids[i] = sTmp.wids[i];
      }
    }
  }
  else if ((FacetsTest[0] + FacetsTest[1] + FacetsTest[2]) == 3)
  {
    /* The origin projections lays onto the triangle */
    double inv_detM = 1 / nu_max;
    s.lambdas[0] = B[2] * inv_detM;
    s.lambdas[1] = B[1] * inv_detM;
    s.lambdas[2] = 1 - s.lambdas[0] - s.lambdas[1];
    s.wids[0] = 0;
    s.wids[1] = 1;
    s.wids[2] = 2;
    s.nvrtx = 3;
  }
  else if (FacetsTest[2] == 0)
  {
    /* The origin projection P faces the segment AB */
    s.nvrtx = 2;
    s.vrtx.row(0) = s.vrtx.row(1);
    s.vrtx.row(1) = s.vrtx.row(2);
    v = S1D(s);
  }
  else if (FacetsTest[1] == 0)
  {
    /* The origin projection P faces the segment AC */
    s.nvrtx = 2;
    s.vrtx.row(1) = s.vrtx.row(2);
    v = S1D(s);
    for (int i = 1; i < s.nvrtx; ++i)
      ++s.wids[i];
  }
  else
  {
    /* The origin projection P faces the segment BC */
    s.nvrtx = 2;
    v = S1D(s);
  }

  vv.setZero();
  for (int i = 0; i < s.nvrtx; ++i)
    vv += s.lambdas[i] * s.vrtx.row(i);
}
//-------------------------------------------------------

/**
 * @brief Finds point of minimum norm of 3-simplex. Robust, but slower, version
 * of algorithm presented in paper.
 */
inline static void S3D(Simplex& s, Eigen::Vector3d& vv)
{
  int FacetsTest[4] = {1, 1, 1, 1};
  int TrianglesToTest[9] = {3, 3, 3, 1, 2, 2, 0, 0, 1};
  int nclosestS = 0;
  int firstaux = 0;
  int secondaux = 0;
  int Flag_sAuxused = 0;
  Eigen::Vector3d v, vtmp;
  Eigen::Vector4d B;
  double tmplamda[4] = {0, 0, 0, 0};
  double sqdist_tmp = 0;
  double detM;
  double inv_detM;
#ifdef ADAPTIVEFP
  double o[3] = {0};
#endif

  Eigen::Vector3d a = s.vrtx.row(3);
  Eigen::Vector3d b = s.vrtx.row(2);
  Eigen::Vector3d c = s.vrtx.row(1);
  Eigen::Vector3d d = s.vrtx.row(0);

  // Eigen::Matrix3d mtmp;
  // mtmp.row(0) = s.vrtx.row(0);
  // mtmp.row(1) = s.vrtx.row(2);
  // mtmp.row(2) = s.vrtx.row(3);

  // B[3] = -s.vrtx.bottomRows(3).determinant();
  // B[2] = mtmp.determinant();
  // mtmp.row(1) = s.vrtx.row(1);
  // B[1] = -mtmp.determinant();
  // B[0] = s.vrtx.topRows(3).determinant();

#ifndef ADAPTIVEFP
  B[0] = -1
         * (b[0] * c[1] * d[2] + b[1] * c[2] * d[0] + b[2] * c[0] * d[1]
            - b[2] * c[1] * d[0] - b[1] * c[0] * d[2] - b[0] * c[2] * d[1]);
  B[1] = +1
         * (a[0] * c[1] * d[2] + a[1] * c[2] * d[0] + a[2] * c[0] * d[1]
            - a[2] * c[1] * d[0] - a[1] * c[0] * d[2] - a[0] * c[2] * d[1]);
  B[2] = -1
         * (a[0] * b[1] * d[2] + a[1] * b[2] * d[0] + a[2] * b[0] * d[1]
            - a[2] * b[1] * d[0] - a[1] * b[0] * d[2] - a[0] * b[2] * d[1]);
  B[3] = +1
         * (a[0] * b[1] * c[2] + a[1] * b[2] * c[0] + a[2] * b[0] * c[1]
            - a[2] * b[1] * c[0] - a[1] * b[0] * c[2] - a[0] * b[2] * c[1]);

  detM = B.sum();
#else
  B[0] = orient3d(o, b, c, d);
  B[1] = orient3d(a, o, c, d);
  B[2] = orient3d(a, b, o, d);
  B[3] = orient3d(a, b, c, o);
  detM = orient3d(a, b, c, d);
#endif

  /* Test if sign of ABCD is equal to the signes of the auxiliary simplices */
  double eps = 1e-13;
  if (fabs(detM) < eps)
  {
    if (fabs(B[2]) < eps && fabs(B[3]) < eps)
      FacetsTest[1] = 0; /* A = B. Test only ACD */
    else if (fabs(B[1]) < eps && fabs(B[3]) < eps)
      FacetsTest[2] = 0; /* A = C. Test only ABD */
    else if (fabs(B[1]) < eps && fabs(B[2]) < eps)
      FacetsTest[3] = 0; /* A = D. Test only ABC */
    else if (fabs(B[0]) < eps && fabs(B[3]) < eps)
      FacetsTest[1] = 0; /* B = C. Test only ACD */
    else if (fabs(B[0]) < eps && fabs(B[2]) < eps)
      FacetsTest[1] = 0; /* B = D. Test only ABD */
    else if (fabs(B[0]) < eps && fabs(B[1]) < eps)
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
      FacetsTest[i] = SAMESIGN(detM, B[i]);
  }

  /* Compare signed volumes and compute barycentric coordinates */
  if (FacetsTest[0] + FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 4)
  {
    /* All signs are equal, therefore the origin is inside the simplex */
    inv_detM = 1 / detM;
    s.lambdas[3] = B[0] * inv_detM;
    s.lambdas[2] = B[1] * inv_detM;
    s.lambdas[1] = B[2] * inv_detM;
    s.lambdas[0] = 1 - s.lambdas[1] - s.lambdas[2] - s.lambdas[3];
    s.wids[0] = 0;
    s.wids[1] = 1;
    s.wids[2] = 2;
    s.wids[3] = 3;
    s.nvrtx = 4;
  }
  else if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 0)
  {
    /* There are three facets facing the origin  */
    Simplex sTmp;
    sqdist_tmp = std::numeric_limits<double>::max();
    int sID[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
      sTmp.nvrtx = 3;
      /* Assign coordinates to simplex */
      for (int k = 0; k < sTmp.nvrtx; ++k)
      {
        const int vtxid = TrianglesToTest[i + (k * 3)];
        sTmp.vrtx.row(2 - k) = s.vrtx.row(vtxid);
      }
      /* ... and call S2D  */
      S2D(sTmp, v);
      /* ... compute aux squared distance */
      vtmp.setZero();
      for (int l = 0; l < sTmp.nvrtx; ++l)
        vtmp += sTmp.lambdas[l] * sTmp.vrtx.row(l);

      if (vtmp.squaredNorm() < sqdist_tmp)
      {
        sqdist_tmp = vtmp.squaredNorm();
        nclosestS = sTmp.nvrtx;
        for (int l = 0; l < nclosestS; ++l)
        {
          sID[l] = TrianglesToTest[i + (sTmp.wids[l] * 3)];
          tmplamda[l] = sTmp.lambdas[l];
        }
      }
    }

    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> tmpscoo1 = s.vrtx;

    /* Store closest simplex */
    s.nvrtx = nclosestS;
    for (int i = 0; i < s.nvrtx; ++i)
    {
      s.vrtx.row(nclosestS - 1 - i) = tmpscoo1.row(sID[i]);
      s.lambdas[i] = tmplamda[i];
      s.wids[nclosestS - 1 - i] = sID[i];
    }
  }
  else if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 1)
  {
    /* There are two   facets facing the origin, need to find the closest.   */
    Simplex sTmp;
    sTmp.nvrtx = 3;

    if (FacetsTest[1] == 0)
    {
      /* ... Test facet ACD  */
      sTmp.vrtx.row(0) = s.vrtx.row(0);
      sTmp.vrtx.row(1) = s.vrtx.row(1);
      sTmp.vrtx.row(2) = s.vrtx.row(3);
      /* ... and call S2D  */
      S2D(sTmp, v);
      /* ... compute aux squared distance */
      vtmp.setZero();
      for (int i = 0; i < sTmp.nvrtx; ++i)
        vtmp += sTmp.lambdas[i] * sTmp.vrtx.row(i);

      sqdist_tmp = vtmp.squaredNorm();
      Flag_sAuxused = 1;
      firstaux = 0;
    }
    if (FacetsTest[2] == 0)
    {
      /* ... Test facet ABD  */
      if (Flag_sAuxused == 0)
      {
        sTmp.vrtx.row(0) = s.vrtx.row(0);
        sTmp.vrtx.row(1) = s.vrtx.row(2);
        sTmp.vrtx.row(2) = s.vrtx.row(3);
        /* ... and call S2D  */
        S2D(sTmp, v);
        /* ... compute aux squared distance */

        vtmp.setZero();
        for (int i = 0; i < sTmp.nvrtx; ++i)
          vtmp += sTmp.lambdas[i] * sTmp.vrtx.row(i);

        sqdist_tmp = vtmp.squaredNorm();
        firstaux = 1;
      }
      else
      {
        s.nvrtx = 3;
        s.vrtx.row(1) = s.vrtx.row(2);
        s.vrtx.row(2) = s.vrtx.row(3);
        /* ... and call S2D itself */
        S2D(s, v);
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
      S2D(s, v);
      secondaux = 2;
    }
    /* Compare outcomes */
    v.setZero();
    for (int i = 0; i < s.nvrtx; ++i)
      v += s.lambdas[i] * s.vrtx.row(i);

    if (v.squaredNorm() < sqdist_tmp)
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
      for (int i = 0; i < s.nvrtx; ++i)
      {
        s.vrtx.row(i) = sTmp.vrtx.row(i);
        s.lambdas[i] = sTmp.lambdas[i];
        s.wids[sTmp.nvrtx - 1 - i]
            = TrianglesToTest[firstaux + (sTmp.wids[i] * 3)];
      }
    }
  }
  else if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 2)
  {
    /* Only one facet is facing the origin */
    if (FacetsTest[1] == 0)
    {
      /* The origin projection P faces the facet ACD */
      s.nvrtx = 3;
      s.vrtx.row(1) = s.vrtx.row(1);
      s.vrtx.row(2) = s.vrtx.row(3);
      S2D(s, v);
    }
    else if (FacetsTest[2] == 0)
    {
      /* The origin projection P faces the facet ABD */
      s.nvrtx = 3;
      s.vrtx.row(1) = s.vrtx.row(2);
      s.vrtx.row(2) = s.vrtx.row(3);
      S2D(s, v);
      /* Keep simplex. Need to update sID only*/
      for (int i = 2; i < s.nvrtx; ++i)
      {
        /* Assume that vertex a is always included in sID. */
        ++s.wids[i];
      }
    }
    else if (FacetsTest[3] == 0)
    {
      /* The origin projection P faces the facet ABC */
      s.nvrtx = 3;
      s.vrtx.row(0) = s.vrtx.row(1);
      s.vrtx.row(1) = s.vrtx.row(2);
      s.vrtx.row(2) = s.vrtx.row(3);
      S2D(s, v);
    }
  }
  else
  {
    /* The origin projection P faces the facet BCD */
    s.nvrtx = 3;
    /* ... and call S2D   */
    S2D(s, v);
    /* Keep simplex. Need to update sID only*/
    for (int i = 0; i < s.nvrtx; ++i)
    {
      /* Assume that vertex a is always included in sID. */
      ++s.wids[i];
    }
  }

  vv.setZero();
  for (int i = 0; i < s.nvrtx; ++i)
    vv += s.lambdas[i] * s.vrtx.row(i);
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
void subalgorithm(Simplex& s, Eigen::Vector3d& v)
{
  switch (s.nvrtx)
  {
  case 4:
    S3D(s, v);
    break;
  case 3:
    S2D(s, v);
    break;
  case 2:
    v = S1D(s);
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

#ifdef ADAPTIVEFP
  exactinit();
#endif

  double eps_rel2 = eps_rel * eps_rel;

  /* Initialise  */
  s.nvrtx = 1;
  Eigen::Vector3d bd1s = bd1.row(0);
  Eigen::Vector3d bd2s = bd2.row(0);
  Eigen::Vector3d v = bd1s - bd2s;
  s.vrtx.row(0) = v;
  // s.p.row(0) = bd1.s;
  // s.q.row(0) = bd2.s;

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
    subalgorithm(s, v);

    // s.p.row(s.nvrtx - 1) = bd1.s;
    // s.q.row(s.nvrtx - 1) = bd2.s;

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