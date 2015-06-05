
AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this		
AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])

# Setting the prefix to the default if only --with-cuda was given
if test "$cuda_prefix" == "yes"; then
	if test "$withval" == "yes"; then
		cuda_prefix="/usr/local/cuda"
	fi
fi
    dnl Provide --with-cuda-lib=<dir> and --with-cuda-include=<dir>.
    AC_ARG_WITH([cuda-lib],
        [AS_HELP_STRING([--with-cuda-lib=<dir>],
                        [Path where CUDA BLAS libraries are installed (default: /usr/local/cuda/lib64)])],
        [CUDA_LIB="$withval"], [CUDA_LIB="/usr/local/cuda/lib64"]
    ) dnl AC_ARG_WITH
    AC_ARG_WITH([cuda-include],
        [AS_HELP_STRING([--with-cuda-include=<dir>],
                        [Path where CUDA BLAS includes are installed (default: /usr/local/cuda/include)])],
        [CUDA_INCLUDE="$withval"], [CUDA_INCLUDE="/usr/local/cuda/include"]
    ) dnl AC_ARG_WITH

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
		CUDA_LDFLAGS="-L$CUDA_LIBDIR -lcudart -lnppc -lnppi -lnpps -lcublas -lcufft"
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

AM_CONDITIONAL([HAVE_CUDA], [test "$enable_cuda" = "yes"])

])
