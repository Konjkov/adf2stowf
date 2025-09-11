from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "stowfn_cpp",
        ["stowfn.cpp"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="stowfn_cpp",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
