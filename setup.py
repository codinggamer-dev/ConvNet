"""Setup script for ConvNet-NumPy package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read the LICENSE file
license_file = Path(__file__).parent / "LICENSE.md"
license_text = "MIT" if license_file.exists() else ""

setup(
    name="convnet",
    version="1.0.0-beta",
    author="codinggamer-dev",
    author_email="ege.tba1940@gmail.com",
    description="A minimal, educational convolutional neural network framework built from scratch using NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codinggamer-dev/ConvNet-NumPy",
    project_urls={
        "Bug Reports": "https://github.com/codinggamer-dev/ConvNet-NumPy/issues",
        "Source": "https://github.com/codinggamer-dev/ConvNet-NumPy",
    },
    packages=find_packages(exclude=["examples", "examples.*", "tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "numba>=0.56.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "cuda11": ["cupy-cuda11x>=10.0.0"],
        "cuda12": ["cupy-cuda12x>=12.0.0"],
        "cuda13": ["cupy-cuda13x>=13.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
    keywords="deep-learning neural-network cnn convolutional numpy education cuda scratch python",
    package_data={
        "convnet": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
)
