import numpy
import os, os.path, tempfile, subprocess, shutil
from setuptools import setup, Extension
from Cython.Build import build_ext, cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True


def checkOpenmpSupport():
    """Adapted from https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script"""
    ompTest = r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Thread %d, Total number of threads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r"test.c"
    with open(filename, "w") as file:
        file.write(ompTest)
    with open(os.devnull, "w") as fnull:
        result = subprocess.call(
            ["cc", "-fopenmp", filename], stdout=fnull, stderr=fnull
        )

    os.chdir(curdir)
    shutil.rmtree(tmpdir)
    if result == 0:
        return True
    else:
        return False


if checkOpenmpSupport() == True:
    ompArgs = ["-fopenmp"]
else:
    ompArgs = None

extensions = [
    Extension(
        "ising_cython",
        sources=["ising_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs,
        libraries=[],
    ),  # Unix-like specific
]

setup(
    #   cmdclass = {'build_ext': build_ext},
    #   ext_modules = extensions
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"})
)
