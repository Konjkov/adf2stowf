from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "stowfn_norm",
        ["stowfn_norm.cpp"],
        extra_compile_args=["-O3"],
    ),
    Pybind11Extension(
        "stowfn_atorbs",
        ["stowfn_atorbs.cpp"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="stowfn_norm",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
