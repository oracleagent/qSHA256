#!/usr/bin/env python3
"""
Setup script for qSHA256.

This script helps set up the environment for running the quantum SHA-256 demo.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    dependencies = [
        'qiskit>=0.45.0',
        'numpy>=1.21.0', 
        'matplotlib>=3.5.0',
        'scipy>=1.7.0'
    ]
    
    print("Installing dependencies...")
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    
    return True

def test_installation():
    """Test that the installation works."""
    print("\nTesting installation...")
    
    try:
        # Test imports
        import qiskit
        print("✓ Qiskit imported successfully")
        
        import numpy
        print("✓ NumPy imported successfully")
        
        import matplotlib
        print("✓ Matplotlib imported successfully")
        
        import scipy
        print("✓ SciPy imported successfully")
        
        # Test our package
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from qsha256 import QuantumSHA256
        print("✓ qSHA256 package imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=== qSHA256 Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now run the demo:")
    print("  python examples/demo.py")

if __name__ == "__main__":
    main()
