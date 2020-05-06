
#include "openGJK.hpp"
#include <chrono>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(opengjk, m)
{
  m.def(
      "gjk_timed",
      [](const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr1,
         const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr2)
          -> std::pair<double, double> {
        auto tstart = std::chrono::system_clock::now();
        double d = gjk(arr1, arr2).norm();
        auto tend = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = tend - tstart;
        return {diff.count(), d};
      },
      "Return tuple pair of runtime and computed GJK distance between bodies");
  m.def(
      "gjk_vector",
      [](const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr1,
         const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr2)
          -> Eigen::Vector3d { return gjk(arr1, arr2); },
      "Compute GJK separation vector between bodies");
  m.def(
      "gjk_distance",
      [](const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr1,
         const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr2)
          -> double { return gjk(arr1, arr2).norm(); },
      "Compute GJK distance between bodies");
}