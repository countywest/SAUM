cuda_inc=#cuda include
cuda_lib=#cuda library
nvcc=#nvcc
tf_inc=#tensorflow include
tf_lib=#tensorflow library

all: tf_nndistance_so.so tf_approxmatch_so.so

tf_nndistance.cu.o: tf_nndistance.cu
	$(nvcc) tf_nndistance.cu -o tf_nndistance.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_nndistance_so.so: tf_nndistance.cpp tf_nndistance.cu.o
	g++ tf_nndistance.cpp tf_nndistance.cu.o -o tf_nndistance_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -std=c++11 -fPIC -O2

tf_approxmatch.cu.o: tf_approxmatch.cu
	$(nvcc) tf_approxmatch.cu -o tf_approxmatch.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_approxmatch_so.so: tf_approxmatch.cpp tf_approxmatch.cu.o
	g++ -shared $(CPPFLAGS) tf_approxmatch.cpp tf_approxmatch.cu.o -o tf_approxmatch_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -std=c++11 -fPIC -O2

clean:
	rm -rf *.o *.so
