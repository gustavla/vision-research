from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import extension
import numpy as np

setup(
    name = "gv",
    ext_modules = cythonize('gv/*.pyx'), # accepts a glob pattern
    include_dirs = [np.get_include()],
)
