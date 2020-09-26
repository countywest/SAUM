#!/usr/bin/env bash
#/bin/bash
cuda_inc=#cuda include
cuda_lib=#cuda library
nvcc=#nvcc
tf_inc=#tensorflow include
tf_inc_pub=#tensorflow include public
tf_lib=#tensorflow library

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc_pub -lcudart -L $cuda_lib -L $tf_lib -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
