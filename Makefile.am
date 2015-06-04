bin_PROGRAMS = mediametrics
mediametrics_SOURCES = src/mediametrics.cpp

mediametrics_CFLAGS=-DMYDEBUG=0 `pkg-config --cflags opencv` #$(OPENCV_CFLAGS)

mediametrics_CPPFLAGS=-DMYDEBUG=0 `pkg-config --cflags opencv` #$(OPENCV_CFLAGS)

mediametrics_LDFLAGS= `pkg-config --libs opencv` -lopencv_gpu `pkg-config --libs x264`


if USE_GPU
DEFS+= -DUSE_GPU=1
endif

if DEBUG
# use "valgrind --tool=memcheck --leak-check=yes" to check memory leak, MemWatch will drag the program.
#DEFS+=-DMEMWATCH=1
DEFS+=-DDEBUG=1
endif
