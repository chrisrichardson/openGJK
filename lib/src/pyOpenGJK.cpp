
#include "openGJK.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(opengjk, m)
{
  m.def("gjk",
        [](const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr1,
           const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& arr2)
            -> double { return gjk(arr1, arr2); });
}