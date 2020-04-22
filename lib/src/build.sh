#!/bin/bash
g++ -Wall -fPIC -shared `python3 -m pybind11 --includes` -I/usr/include/eigen3 openGJK.cpp -o opengjk`python3-config --extension-suffix`
g++ -Wall -fPIC -shared `python3 -m pybind11 --includes` -I/usr/include/eigen3 openGJK.c -o opengjkc`python3-config --extension-suffix`

# g++ -Wall -fPIC -shared -I/usr/include/python8 -I/home/fenics/.local/include -I../include -I/usr/include/eigen3 openGJK.c -o opengjkc`python3-config --extension-suffix`
