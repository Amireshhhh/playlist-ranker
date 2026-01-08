"""Setup script for playlist-ranker package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="playlist-ranker",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Constraint-aware playlist ranking system using Spotify-scale data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/playlist-ranker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.24.0",
            "ipywidgets>=8.0.7",
        ],
        "mlops": [
            "mlflow>=2.5.0",
            "wandb>=0.15.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "playlist-ranker=serving.api:main",
        ],
    },
)
