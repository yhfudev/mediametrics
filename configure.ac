#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT(mediamatrics, 0.2.0, yhfudev@gmail.com)
AC_CONFIG_SRCDIR([src/mediametrics.cpp])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.10 -Wall no-define])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CPP
AC_PROG_INSTALL

# Checks for libraries.
PKG_CHECK_MODULES(OPENCV,opencv,HAVE_OPENCV=yes, HAVE_OPENCV=no)

AC_SUBST(OPENCV_CFLAGS)
AC_SUBST(OPENCV_LIBS)

if test "x$HAVE_OPENCV" = "xno"; then
  AC_MSG_ERROR(OpenCV libraries could not be found)
fi

# Checks for header files.
AC_CHECK_HEADERS([string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
#AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([pow strchr strstr])

AC_CONFIG_FILES([Makefile])

AC_OUTPUT

AC_MSG_RESULT([
General configuration
Compiler: ${CXX}
CXXFLAGS: ${CXXFLAGS}

Install path: ${prefix}

Now run make ...
])
