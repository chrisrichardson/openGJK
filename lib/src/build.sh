#!/bin/bash
set -x
# g++ -Wall -fPIC -shared `python3 -m pybind11 --includes` -I/usr/include/eigen3 openGJK.cpp -o opengjk`python3-config --extension-suffix`

g++ -Wall -fPIC -shared `python3 -m pybind11 --includes` -I../include -I/usr/include/eigen3 openGJK.c -o opengjkc`python3-config --extension-suffix`
