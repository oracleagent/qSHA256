#!/usr/bin/env python3
"""
Setup script for qSHA256 - A cryptographically secure cryptographic library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qsha256",
    version="1.0.0",
    author="qSHA256 Contributors",
    author_email="security@qsha256.org",
    description="A cryptographically secure cryptographic library with SHA-256, HMAC, AES-GCM, Ed25519, and HKDF",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/oracleagent/qSHA256",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=[
        "cryptography>=41.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
        "demo": [
            "qiskit>=0.45.0",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "scipy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qsha256-demo=demos.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="cryptography, security, sha256, hmac, aes-gcm, ed25519, hkdf, encryption, authentication, digital-signatures",
    project_urls={
        "Bug Reports": "https://github.com/oracleagent/qSHA256/issues",
        "Source": "https://github.com/oracleagent/qSHA256",
        "Security": "https://github.com/oracleagent/qSHA256/blob/main/SECURITY.md",
        "Documentation": "https://github.com/oracleagent/qSHA256/blob/main/README.md",
    },
)