from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import extension

setup(
    name = "gv",
    ext_modules = cythonize('gv/*.pyx'), # accepts a glob pattern
)
