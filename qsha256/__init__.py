"""
qSHA256 - A quantum implementation of SHA-256 hash function components.

This package demonstrates how to implement SHA-256-like operations on quantum circuits
using Qiskit. This is for educational purposes only and not suitable for production cryptography.
"""

__version__ = "1.0.0"
__author__ = "qSHA256 Contributors"

from .core import quantum_xor, quantum_and, quantum_ch, quantum_sigma0, quantum_sigma1, quantum_maj
from .hash import QuantumSHA256
from .utils import calculate_entropy, calculate_coherence, analyze_state_evolution

__all__ = [
    "quantum_xor",
    "quantum_and", 
    "quantum_ch",
    "quantum_sigma0",
    "quantum_sigma1",
    "quantum_maj",
    "QuantumSHA256",
    "calculate_entropy",
    "calculate_coherence",
    "analyze_state_evolution"
]
