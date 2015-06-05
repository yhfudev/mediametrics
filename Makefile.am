bin_PROGRAMS = mediametrics
mediametrics_SOURCES = src/mediametrics.cpp

#mediametrics_CFLAGS=-DMYDEBUG=0 `pkg-config --cflags opencv` #$(OPENCV_CFLAGS)
#mediametrics_CPPFLAGS=-DMYDEBUG=0 `pkg-config --cflags opencv` #$(OPENCV_CFLAGS)
#mediametrics_LDFLAGS= `pkg-config --libs opencv` -lopencv_gpu `pkg-config --libs x264`

mediametrics_CFLAGS=$(OPENCV_CFLAGS) @CUDA_CFLAGS@
mediametrics_CPPFLAGS=$(OPENCV_CFLAGS) $(CUDA_CFLAGS)
mediametrics_LDFLAGS=$(OPENCV_LIBS) $(CUDA_LDFLAGS)


if HAVE_CUDA
DEFS+= -DUSE_CUDA=1
endif

if DEBUG
# use "valgrind --tool=memcheck --leak-check=yes" to check memory leak, MemWatch will drag the program.
#DEFS+=-DMEMWATCH=1
DEFS+=-DDEBUG=1
endif


# CUDA, you need also add the file ax_check_cuda.m4 to your autoconf folder
# rules for CUDA:
.cu.o: xraylib-cuda.h xraylib-cuda-private.h
	$(NVCC) $(libxrlcuda_a_CFLAGS) -arch=sm_20 -dc -o $@ $<

#.cu.lo:
#	$(LIBTOOL) --tag=CXX --mode=compile $(NVCCF) $(NVCC_OPTIONS) -DHAVE_CONFIG_H --compile -o $@ $^ -shared "--compiler-options -fPIC -DPIC" -I../include -I../..
