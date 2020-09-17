from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "Something",                  # Not the name of the module
    cmdclass = {"build_ext":build_ext},  # magic
    ext_modules = [Extension("protien", # The name of the module
            ["cpaf.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-fopenmp'],
            extra_link_args=["-O3", '-fopenmp'],
            libraries=["pafomp", "gomp"])]      # <lib>scalars<.a>
)
