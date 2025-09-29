"""
EXPERIMENTAL QUANTUM SHA-256 DEMONSTRATIONS - NOT SECURE

⚠️  WARNING: This module contains experimental quantum implementations of SHA-256
    components for educational and research purposes only.

⚠️  DO NOT USE FOR PRODUCTION CRYPTOGRAPHY - These implementations are not
    cryptographically secure and should never be used in real applications.

⚠️  FOR DEMONSTRATION ONLY - These demos show quantum circuit implementations
    of SHA-256 operations using Qiskit, but they are not suitable for any
    security-critical applications.

This module is completely separate from the main qSHA256 secure library.
Use qsha256.secure for production cryptographic operations.
"""

# Prevent accidental imports in production code
import warnings

warnings.warn(
    "You are importing experimental quantum SHA-256 demos. "
    "These are NOT SECURE and should only be used for educational purposes. "
    "Use qsha256.secure for production cryptography.",
    UserWarning,
    stacklevel=2
)
