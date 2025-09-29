"""
qSHA256 - A cryptographically secure cryptographic library.

This package provides secure implementations of cryptographic primitives using
vetted libraries (hashlib, cryptography) for production use.
"""

__version__ = "1.0.0"
__author__ = "qSHA256 Contributors"

from .secure import (
    secure_sha256,
    secure_hmac,
    secure_hmac_verify,
    generate_key,
    hkdf_extract_expand,
    aes_gcm_encrypt,
    aes_gcm_decrypt,
    ed25519_generate_keypair,
    ed25519_sign,
    ed25519_verify,
    SecurityError
)

__all__ = [
    "secure_sha256",
    "secure_hmac",
    "secure_hmac_verify", 
    "generate_key",
    "hkdf_extract_expand",
    "aes_gcm_encrypt",
    "aes_gcm_decrypt",
    "ed25519_generate_keypair",
    "ed25519_sign",
    "ed25519_verify",
    "SecurityError"
]
