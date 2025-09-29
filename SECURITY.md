# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Design

qSHA256 is designed with security as a primary concern:

### Cryptographic Foundations
- **Vetted Libraries**: All cryptographic operations use well-established libraries (`hashlib`, `cryptography`, `hmac`)
- **No Custom Crypto**: We do not implement custom cryptographic algorithms
- **Secure by Default**: All functions include proper input validation and secure defaults

### Security Features
- **Input Validation**: Strict type checking and size limits (1 MB default)
- **Constant-Time Operations**: HMAC verification uses constant-time comparison to prevent timing attacks
- **Secure Random**: Key generation uses cryptographically secure random number generation
- **Error Handling**: Secure error handling that doesn't leak sensitive information

### Audit Recommendations

While qSHA256 is built on vetted cryptographic libraries, we recommend:

1. **Independent Security Audits**: Before using in production environments handling sensitive data
2. **Regular Updates**: Keep dependencies updated to latest secure versions
3. **Security Testing**: Include qSHA256 in your security testing procedures

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in qSHA256, please follow these steps:

### How to Report

1. **Do NOT** create a public GitHub issue
2. Email security details to: `security@qsha256.org` (placeholder - update with actual contact)
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Within 30 days (or coordinated disclosure timeline)

### Disclosure Process

1. **Confirmation**: We'll confirm receipt of your report
2. **Investigation**: We'll investigate and verify the vulnerability
3. **Fix Development**: We'll develop and test a fix
4. **Coordinated Disclosure**: We'll coordinate public disclosure with you
5. **Release**: We'll release a security update

### What We Expect

- **Responsible Disclosure**: Please allow us reasonable time to address the issue before public disclosure
- **Good Faith**: Report vulnerabilities in good faith without attempting to access data beyond what's necessary
- **No Malicious Use**: Do not use the vulnerability for malicious purposes

### What You Can Expect

- **Credit**: We'll credit you in security advisories (unless you prefer to remain anonymous)
- **Responsiveness**: We'll keep you informed of our progress
- **Recognition**: We'll acknowledge your contribution to improving qSHA256's security

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest version of qSHA256
2. **Validate Inputs**: Validate all inputs before passing to qSHA256 functions
3. **Secure Key Management**: Store cryptographic keys securely
4. **Use HTTPS**: Always use secure channels for key exchange
5. **Regular Audits**: Include qSHA256 in your security audit procedures

### For Developers

1. **Input Validation**: Always validate inputs before cryptographic operations
2. **Error Handling**: Implement proper error handling without information leakage
3. **Key Management**: Follow secure key management practices
4. **Testing**: Include security testing in your development process

## Known Security Considerations

### Input Size Limits
- Default maximum input size is 1 MB to prevent memory exhaustion attacks
- This can be configured but should be carefully considered

### Key Management
- qSHA256 provides secure key generation but not key storage
- Implementers must handle key storage securely

### Random Number Generation
- All random operations use `secrets.token_bytes()` for cryptographic security
- Ensure your system has sufficient entropy

## Security Contacts

- **Security Email**: `security@qsha256.org` (placeholder - update with actual contact)
- **Maintainer**: qSHA256 Contributors
- **GitHub Issues**: Use for non-security bugs only

## Security History

### Version 1.0.0
- Initial secure release
- Built on vetted cryptographic libraries
- Comprehensive input validation
- Security-focused design

## Acknowledgments

We thank the security researchers and community members who help make qSHA256 more secure through responsible disclosure and security feedback.

## Legal

This security policy is provided for informational purposes only and does not constitute legal advice. Users are responsible for ensuring their use of qSHA256 complies with applicable laws and regulations.
