"""
Setup script for Active Learning Molecular Docking package
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
init_file = Path(__file__).parent / "__init__.py"
with open(init_file, 'r', encoding='utf-8') as f:
    content = f.read()
    version = re.search(r'__version__ = ["\']([^"\']+)["\']', content).group(1)

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Active Learning framework for molecular docking with TabM deep learning model"

# Read requirements from environment.yml (simplified)
requirements = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "torch>=1.11.0",
    "rdkit-pypi>=2022.3.0",  # PyPI version of RDKit
    "optuna>=2.10.0",
    "joblib>=1.1.0",
    "tqdm>=4.62.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "psutil>=5.8.0",
    "pyyaml>=6.0",
    "jsonschema>=4.0.0",
    "biopython>=1.79",
]

# Optional dependencies for development
dev_requirements = [
    "pytest>=6.2.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "jupyter>=1.0.0",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

# Optional dependencies for GPU support
gpu_requirements = [
    "torch>=1.11.0+cu116",
    "torchvision>=0.12.0+cu116",
    "torchaudio>=0.11.0+cu116",
]

setup(
    name="active-learning-docking",
    version=version,
    author="Active Learning Molecular Docking Team",
    author_email="dyoh@research.institution.edu",
    description="Active learning framework for molecular docking with TabM deep learning model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USERNAME/ActiveLearning_only_tabm",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": gpu_requirements,
        "all": dev_requirements + gpu_requirements,
    },
    entry_points={
        "console_scripts": [
            "active-learning-docking=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.md", "*.txt"],
        "bin": ["*"],
    },
    zip_safe=False,
    keywords=[
        "active learning",
        "molecular docking",
        "machine learning",
        "drug discovery",
        "computational chemistry",
        "deep learning",
        "pytorch",
        "rdkit",
        "autodock vina",
    ],
    project_urls={
        "Bug Reports": "https://github.com/USERNAME/ActiveLearning_only_tabm/issues",
        "Source": "https://github.com/USERNAME/ActiveLearning_only_tabm",
        "Documentation": "https://github.com/USERNAME/ActiveLearning_only_tabm/blob/master/README.md",
    },
)