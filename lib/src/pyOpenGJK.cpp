
#include "openGJK.hpp"
#include <chrono>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(opengjk, m)
{

  m.def("gjk_timed",
        [](const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr1,
           const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr2)
            -> std::pair<double, double> {
          auto tstart = std::chrono::system_clock::now();
          double d = gjk(arr1, arr2);
          auto tend = std::chrono::system_clock::now();
          std::chrono::duration<double> diff = tend - tstart;
          //  std::cout << "OpenGJK time = " << diff.count() * 1e6 << " us\n";
          return {diff.count(), d};
        });
  m.def("gjk",
        [](const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr1,
           const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr2)
            -> double { return gjk(arr1, arr2); });
}