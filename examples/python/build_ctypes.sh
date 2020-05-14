#!/bin/bash
gcc -shared -Wall -fPIC -fopenmp -shared  -I../../include ../../src/openGJK.c -o opengjk-ctypes.so