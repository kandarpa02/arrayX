from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os

ext_modules = cythonize([
    Extension(
        name="neo._src.autograd.GRAPH_MANAGER",
        sources=["neo/_src/autograd/GRAPH_MANAGER.pyx"],
        language="c++",
    )
])

setup(
    name="neonet",
    version="0.0.1a1",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="An autodiff library for personal use",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/neonet.git",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache-2.0",
    zip_safe=False,
)
