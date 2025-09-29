# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- **Initial secure release** - Complete refactor from experimental quantum implementation to production-ready cryptographic library
- **SHA-256 Hashing**: `secure_sha256()` function using Python's `hashlib`
- **HMAC Authentication**: `secure_hmac()` and `secure_hmac_verify()` with constant-time comparison
- **Key Generation**: `generate_key()` for cryptographically secure random key generation
- **HKDF Key Derivation**: `hkdf_extract_expand()` for secure key material extraction and expansion
- **AES-GCM Encryption**: `aes_gcm_encrypt()` and `aes_gcm_decrypt()` for authenticated encryption
- **Ed25519 Signatures**: `ed25519_generate_keypair()`, `ed25519_sign()`, and `ed25519_verify()` for digital signatures
- **Input Validation**: Strict type checking and configurable size limits (default: 1 MB)
- **Security Error Handling**: Custom `SecurityError` exception for security-related failures
- **Comprehensive Test Suite**: Full test coverage for all cryptographic functions
- **CI/CD Pipeline**: GitHub Actions workflow with Python 3.10/3.11 matrix testing
- **Security Scanning**: Integrated `bandit` and `safety` for security vulnerability detection
- **Code Quality**: `ruff` linting and formatting
- **Documentation**: Complete API documentation with usage examples
- **Security Policy**: `SECURITY.md` with vulnerability disclosure guidelines

### Changed
- **Complete API Overhaul**: Replaced all experimental quantum functions with secure cryptographic primitives
- **Library Focus**: Shifted from educational quantum computing to production cryptography
- **Dependencies**: Updated from Qiskit-based to cryptography-based dependencies
- **Package Structure**: Moved experimental demos to separate `demos/` folder with security warnings

### Security
- **Vetted Libraries**: All cryptographic operations now use well-established libraries (`hashlib`, `cryptography`, `hmac`)
- **No Custom Crypto**: Removed all custom cryptographic implementations
- **Secure by Default**: All functions include proper input validation and secure defaults
- **Constant-Time Operations**: HMAC verification uses constant-time comparison
- **Secure Random**: Key generation uses `secrets.token_bytes()` for cryptographic security
- **Input Sanitization**: Strict validation of input types and sizes

### Removed
- **Experimental Quantum Functions**: All quantum circuit implementations moved to `demos/` folder
- **Qiskit Dependencies**: Removed quantum computing framework dependencies from main package
- **Educational-Only Warnings**: Replaced with production-ready security focus

### Deprecated
- **Experimental Modules**: All quantum SHA-256 implementations in `demos/` are marked as "NOT SECURE - FOR DEMONSTRATION ONLY"

### Fixed
- **Security Vulnerabilities**: Addressed all potential security issues in experimental implementation
- **Input Validation**: Added comprehensive input validation to prevent injection and overflow attacks
- **Error Handling**: Improved error handling to prevent information leakage

### Technical Details
- **Python Support**: Python 3.10+ required
- **Dependencies**: `cryptography>=41.0.0` for core cryptographic operations
- **Testing**: `pytest>=7.0.0` with comprehensive test coverage
- **Linting**: `ruff>=0.1.0` for code quality
- **Security**: `bandit>=1.7.0` and `safety>=2.3.0` for security scanning

## Pre-1.0.0 (Historical)

### [0.x.x] - Previous Versions
- **Experimental Quantum Implementation**: Educational quantum SHA-256 using Qiskit
- **Quantum Gates**: XOR, AND, Ch, Maj, Sigma functions implemented as quantum circuits
- **State Analysis**: Quantum state metrics and visualization
- **Educational Focus**: Designed for learning quantum computing concepts

**Note**: All pre-1.0.0 versions contained experimental quantum implementations that were **NOT SECURE** and should never be used for production cryptography. These implementations have been moved to the `demos/` folder with appropriate security warnings.

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

## Security Notes

- All versions 1.0.0+ are designed for production use with secure cryptographic primitives
- Versions 0.x.x were experimental and educational only - **DO NOT USE FOR PRODUCTION**
- Always use the latest version for security updates
- Report security vulnerabilities according to [SECURITY.md](SECURITY.md)
