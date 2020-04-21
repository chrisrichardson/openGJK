# Set pybind11_DIR to point to /somepath/lib/python3.x/site-packages/pybind11
# Set CPATH to include /somepath/include/eigen3

cmake_minimum_required(VERSION 2.8.12)
project(opengjk)

find_package(pybind11)
pybind11_add_module(opengjk NO_EXTRAS openGJK.cpp)
