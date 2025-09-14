from setuptools import setup, find_packages, Extension

# Setup configuration
setup(
    name="ArrayX",
    version="0.0.1a1",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="An autodiff library for personal use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/ArrayX.git",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
    zip_safe=False,
)
