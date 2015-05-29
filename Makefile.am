bin_PROGRAMS = mediametrics
mediametrics_SOURCES = src/mediametrics.cpp

mediametrics_CFLAGS=-DUSEGPU=1 -DMYDEBUG=0 `pkg-config --cflags opencv` #$(OPENCV_CFLAGS)

mediametrics_CPPFLAGS=-DUSEGPU=1 -DMYDEBUG=0 `pkg-config --cflags opencv` #$(OPENCV_CFLAGS)

#mediametrics_LDADD=$(OPENCV_LIBS)
mediametrics_LDADD=`pkg-config --libs opencv`
