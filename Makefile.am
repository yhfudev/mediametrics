bin_PROGRAMS = mediametrics
mediametrics_SOURCES = src/mediametrics.cpp

mediametrics_CFLAGS=-DUSEGPU=1 -DMYDEBUG=0 $(OPENCV_CFLAGS)

mediametrics_CPPFLAGS=-DUSEGPU=1 -DMYDEBUG=0 $(OPENCV_CFLAGS)

mediametrics_LDADD=$(OPENCV_LIBS)
