"""
Cryptographically secure cryptographic primitives for qSHA256.

This module provides secure implementations of cryptographic functions using
vetted libraries (hashlib, cryptography) rather than custom implementations.
All functions include strict input validation and are designed for production use.
"""

import hashlib
import hmac
import os
import secrets
from typing import Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend


# Configuration
MAX_INPUT_SIZE = 1024 * 1024  # 1 MB default limit
DEFAULT_KEY_SIZE = 32  # 256 bits
DEFAULT_NONCE_SIZE = 12  # 96 bits for AES-GCM


class SecurityError(Exception):
    """Raised when security constraints are violated."""
    pass


def _validate_bytes_input(data: Union[bytes, bytearray], name: str) -> bytes:
    """
    Validate that input is bytes/bytearray and within size limits.
    
    Args:
        data: Input data to validate
        name: Name of the parameter for error messages
        
    Returns:
        bytes: Validated bytes data
        
    Raises:
        TypeError: If data is not bytes or bytearray
        ValueError: If data exceeds size limits
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes or bytearray, got {type(data).__name__}")
    
    if len(data) > MAX_INPUT_SIZE:
        raise ValueError(f"{name} size ({len(data)} bytes) exceeds maximum allowed size ({MAX_INPUT_SIZE} bytes)")
    
    return bytes(data)


def secure_sha256(data: Union[bytes, bytearray]) -> bytes:
    """
    Compute SHA-256 hash using Python's cryptographically secure hashlib.
    
    Args:
        data: Input data to hash
        
    Returns:
        bytes: 32-byte SHA-256 hash
        
    Raises:
        TypeError: If data is not bytes or bytearray
        ValueError: If data exceeds size limits
    """
    data = _validate_bytes_input(data, "data")
    return hashlib.sha256(data).digest()


def secure_hmac(key: Union[bytes, bytearray], data: Union[bytes, bytearray]) -> bytes:
    """
    Compute HMAC using Python's cryptographically secure hmac module.
    
    Args:
        key: HMAC key
        data: Data to authenticate
        
    Returns:
        bytes: HMAC tag (32 bytes for SHA-256)
        
    Raises:
        TypeError: If key or data are not bytes or bytearray
        ValueError: If inputs exceed size limits
    """
    key = _validate_bytes_input(key, "key")
    data = _validate_bytes_input(data, "data")
    
    if len(key) < 16:  # Minimum recommended key size
        raise ValueError("HMAC key must be at least 16 bytes")
    
    return hmac.new(key, data, hashlib.sha256).digest()


def secure_hmac_verify(key: Union[bytes, bytearray], data: Union[bytes, bytearray], tag: Union[bytes, bytearray]) -> bool:
    """
    Verify HMAC tag using constant-time comparison.
    
    Args:
        key: HMAC key
        data: Original data
        tag: HMAC tag to verify
        
    Returns:
        bool: True if tag is valid, False otherwise
        
    Raises:
        TypeError: If inputs are not bytes or bytearray
        ValueError: If inputs exceed size limits
    """
    key = _validate_bytes_input(key, "key")
    data = _validate_bytes_input(data, "data")
    tag = _validate_bytes_input(tag, "tag")
    
    if len(key) < 16:
        raise ValueError("HMAC key must be at least 16 bytes")
    
    expected_tag = hmac.new(key, data, hashlib.sha256).digest()
    return hmac.compare_digest(tag, expected_tag)


def generate_key(length: int = DEFAULT_KEY_SIZE) -> bytes:
    """
    Generate a cryptographically secure random key.
    
    Args:
        length: Key length in bytes (default: 32)
        
    Returns:
        bytes: Cryptographically secure random key
        
    Raises:
        ValueError: If length is invalid
    """
    if length < 1:
        raise ValueError("Key length must be positive")
    if length > 1024:  # Reasonable upper limit
        raise ValueError("Key length too large (max 1024 bytes)")
    
    return secrets.token_bytes(length)


def hkdf_extract_expand(salt: Union[bytes, bytearray], info: Union[bytes, bytearray], 
                       ikm: Union[bytes, bytearray], length: int = DEFAULT_KEY_SIZE) -> bytes:
    """
    Extract and expand key material using HKDF.
    
    Args:
        salt: Salt for key extraction
        info: Context information
        ikm: Input key material
        length: Output key length in bytes
        
    Returns:
        bytes: Derived key material
        
    Raises:
        TypeError: If inputs are not bytes or bytearray
        ValueError: If inputs exceed size limits or length is invalid
    """
    salt = _validate_bytes_input(salt, "salt")
    info = _validate_bytes_input(info, "info")
    ikm = _validate_bytes_input(ikm, "ikm")
    
    if length < 1 or length > 255 * 32:  # HKDF limit
        raise ValueError("Invalid HKDF output length")
    
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
        backend=default_backend()
    )
    return hkdf.derive(ikm)


def aes_gcm_encrypt(key: Union[bytes, bytearray], plaintext: Union[bytes, bytearray], 
                   aad: Union[bytes, bytearray, None] = None) -> tuple[bytes, bytes]:
    """
    Encrypt data using AES-GCM authenticated encryption.
    
    Args:
        key: AES key (must be 16, 24, or 32 bytes)
        plaintext: Data to encrypt
        aad: Additional authenticated data (optional)
        
    Returns:
        tuple: (nonce, ciphertext_and_tag) where nonce is 12 bytes and 
               ciphertext_and_tag contains encrypted data + 16-byte tag
        
    Raises:
        TypeError: If inputs are not bytes or bytearray
        ValueError: If inputs exceed size limits or key is invalid length
    """
    key = _validate_bytes_input(key, "key")
    plaintext = _validate_bytes_input(plaintext, "plaintext")
    
    if len(key) not in (16, 24, 32):
        raise ValueError("AES key must be 16, 24, or 32 bytes")
    
    if aad is not None:
        aad = _validate_bytes_input(aad, "aad")
    
    # Generate random nonce
    nonce = secrets.token_bytes(DEFAULT_NONCE_SIZE)
    
    # Encrypt
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    
    return nonce, ciphertext


def aes_gcm_decrypt(key: Union[bytes, bytearray], nonce: Union[bytes, bytearray], 
                   ct: Union[bytes, bytearray], aad: Union[bytes, bytearray, None] = None) -> bytes:
    """
    Decrypt data using AES-GCM authenticated encryption.
    
    Args:
        key: AES key (must be 16, 24, or 32 bytes)
        nonce: Nonce used for encryption
        ct: Ciphertext and tag
        aad: Additional authenticated data (optional)
        
    Returns:
        bytes: Decrypted plaintext
        
    Raises:
        TypeError: If inputs are not bytes or bytearray
        ValueError: If inputs exceed size limits, key is invalid, or decryption fails
        SecurityError: If authentication fails
    """
    key = _validate_bytes_input(key, "key")
    nonce = _validate_bytes_input(nonce, "nonce")
    ct = _validate_bytes_input(ct, "ct")
    
    if len(key) not in (16, 24, 32):
        raise ValueError("AES key must be 16, 24, or 32 bytes")
    
    if len(nonce) != DEFAULT_NONCE_SIZE:
        raise ValueError(f"Nonce must be {DEFAULT_NONCE_SIZE} bytes")
    
    if aad is not None:
        aad = _validate_bytes_input(aad, "aad")
    
    try:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ct, aad)
        return plaintext
    except Exception as e:
        raise SecurityError(f"Decryption failed: {e}")


def ed25519_generate_keypair() -> tuple[bytes, bytes]:
    """
    Generate an Ed25519 key pair.
    
    Returns:
        tuple: (private_key, public_key) both as 32-byte raw keys
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    # Serialize to raw bytes
    private_raw = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    return private_raw, public_raw


def ed25519_sign(private_key_raw: Union[bytes, bytearray], message: Union[bytes, bytearray]) -> bytes:
    """
    Sign a message using Ed25519.
    
    Args:
        private_key_raw: Raw 32-byte private key
        message: Message to sign
        
    Returns:
        bytes: 64-byte Ed25519 signature
        
    Raises:
        TypeError: If inputs are not bytes or bytearray
        ValueError: If inputs exceed size limits or private key is invalid
    """
    private_key_raw = _validate_bytes_input(private_key_raw, "private_key_raw")
    message = _validate_bytes_input(message, "message")
    
    if len(private_key_raw) != 32:
        raise ValueError("Ed25519 private key must be 32 bytes")
    
    try:
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_raw)
        return private_key.sign(message)
    except Exception as e:
        raise ValueError(f"Invalid Ed25519 private key: {e}")


def ed25519_verify(public_key_raw: Union[bytes, bytearray], message: Union[bytes, bytearray], 
                  signature: Union[bytes, bytearray]) -> bool:
    """
    Verify an Ed25519 signature.
    
    Args:
        public_key_raw: Raw 32-byte public key
        message: Original message
        signature: 64-byte signature to verify
        
    Returns:
        bool: True if signature is valid, False otherwise
        
    Raises:
        TypeError: If inputs are not bytes or bytearray
        ValueError: If inputs exceed size limits or keys are invalid length
    """
    public_key_raw = _validate_bytes_input(public_key_raw, "public_key_raw")
    message = _validate_bytes_input(message, "message")
    signature = _validate_bytes_input(signature, "signature")
    
    if len(public_key_raw) != 32:
        raise ValueError("Ed25519 public key must be 32 bytes")
    
    if len(signature) != 64:
        raise ValueError("Ed25519 signature must be 64 bytes")
    
    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_raw)
        public_key.verify(signature, message)
        return True
    except Exception:
        return False
