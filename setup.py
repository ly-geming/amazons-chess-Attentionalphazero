import sys
from setuptools import setup, Extension
import pybind11

# 根据操作系统选择编译参数
if sys.platform == "win32":
    # Windows / MSVC 参数
    extra_compile_args = ['/O2', '/arch:AVX2', '/std:c++17']
else:
    # Linux / GCC 参数
    extra_compile_args = ['-O3', '-mavx2', '-std=c++17', '-pthread']

ext_modules = [
    Extension(
        "amazons_ops",
        ["amazons_ops.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="amazons_ops",
    version="1.0",
    ext_modules=ext_modules,
)