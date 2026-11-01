"""Setup script for playlist-ranker package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def read_requirements(path):
    """Requirement lines from a requirements file, without comments or -r includes."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = (line.split("#", 1)[0].strip() for line in fh)
        return [line for line in lines if line and not line.startswith("-")]


requirements = read_requirements("requirements.txt")

setup(
    name="playlist-ranker",
    version="1.0.0",
    author="Amirreza Eshraghi",
    description="Constraint-aware playlist ranking system using Spotify-scale data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Amireshhhh/playlist-ranker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "playlist-ranker=serving.api:main",
        ],
    },
)
