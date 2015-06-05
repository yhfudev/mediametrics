#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT(mediamatrics, 0.2.0, yhfudev@gmail.com)
AC_CONFIG_SRCDIR([src/mediametrics.cpp])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([-Wall no-define subdir-objects])

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

# debug
AC_ARG_ENABLE([debug],
	AS_HELP_STRING([--enable-debug],[Compile the debug version (default: disabled)]),
	[enable_debug=$enableval],
	[enable_debug=no])
AM_CONDITIONAL([DEBUG], [test $enable_debug = "yes"])
if test "x$enable_debug" = "xyes"; then
  changequote({,})
  CFLAGS=`echo "$CFLAGS" | $SED -e 's/-O[0-9s]*//g'`
  CXXFLAGS=`echo "$CXXFLAGS" | $SED -e 's/-O[0-9s]*//g'`
  changequote([,])
  dnl add -O0 only if GCC or ICC is used
  if test "$GCC" = "yes" || test "$ICC" = "yes"; then
    CFLAGS="$CFLAGS -g -O0 -Wall"
    CXXFLAGS="$CXXFLAGS -g -O0 -Wall"
  fi
else
  changequote({,})
  CFLAGS=`echo "$CFLAGS" | $SED -e 's/-g//g'`
  CXXFLAGS=`echo "$CXXFLAGS" | $SED -e 's/-g//g'`
  changequote([,])
fi


AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this
AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])

# Setting the prefix to the default if only --with-cuda was given
if test "$cuda_prefix" == "yes"; then
	if test "$withval" == "yes"; then
		cuda_prefix="/usr/local/cuda"
	fi
fi

# Checking for nvcc
AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
if test -x "$cuda_prefix/bin/nvcc"; then
	AC_MSG_RESULT([found])
	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	# We need to add the CUDA search directories for header and lib searches

	CUDA_CFLAGS=""

	# Saving the current flags
	ax_save_CFLAGS="${CFLAGS}"
	ax_save_LDFLAGS="${LDFLAGS}"

	# Announcing the new variables
	AC_SUBST([CUDA_CFLAGS])
	AC_SUBST([CUDA_LDFLAGS])
	AC_SUBST([NVCC],[$cuda_prefix/bin/nvcc])
	AC_CHECK_FILE([$cuda_prefix/lib64],[lib64_found=yes],[lib64_found=no])
	if test "x$lib64_found" = xno ; then
		AC_CHECK_FILE([$cuda_prefix/lib],[lib32_found=yes],[lib32_found=no])
		if test "x$lib32_found" = xyes ; then
			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
		else
			AC_MSG_WARN([Couldn't find cuda lib directory])
			VALID_CUDA=no
		fi
	else
		AC_CHECK_SIZEOF([long])
		if test "x$ac_cv_sizeof_long" = "x8" ; then
			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib64])
			CUDA_CFLAGS+=" -m64"
		elif test "x$ac_cv_sizeof_long" = "x4" ; then
			AC_CHECK_FILE([$cuda_prefix/lib32],[lib32_found=yes],[lib32_found=no])
			if test "x$lib32_found" = xyes ; then
				AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
				CUDA_CFLAGS+=" -m32"
			else
				AC_MSG_WARN([Couldn't find cuda lib directory])
				VALID_CUDA=no
			fi
		else
			AC_MSG_ERROR([Could not determine size of long variable type])
		fi
	fi

	if test "x$VALID_CUDA" != xno ; then
		CUDA_CFLAGS+=" -I$cuda_prefix/include"
		CFLAGS="$CUDA_CFLAGS $CFLAGS"
		CUDA_LDFLAGS="-L$CUDA_LIBDIR"
		LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"

		# And the header and the lib
		AC_CHECK_HEADER([cuda.h], [],
			AC_MSG_WARN([Couldn't find cuda.h])
			VALID_CUDA=no
			,[#include <cuda.h>])
		if test "x$VALID_CUDA" != "xno" ; then
			AC_CHECK_LIB([cuda], [cuInit], [VALID_CUDA=yes], AC_MSG_WARN([Couldn't find libcuda]
			VALID_CUDA=no))
		fi
	fi
	# Returning to the original flags
	CFLAGS=${ax_save_CFLAGS}
	LDFLAGS=${ax_save_LDFLAGS}
else
	AC_MSG_RESULT([not found!])
	AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
	VALID_CUDA=no
fi

if test "x$enable_cuda" = xyes && test x$VALID_CUDA = xyes ; then
	AC_MSG_NOTICE([Building with CUDA bindings])
elif test "x$enable_cuda" = xyes && test x$VALID_CUDA = xno ; then
	AC_MSG_ERROR([Cannot build CUDA bindings. Check errors])
fi
])
AM_CONDITIONAL([USE_CUDA], [test $enable_cuda = "yes"])


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
