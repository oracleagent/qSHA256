
# QuantumSHA256

## Overview
QuantumSHA256 is an experimental repository aimed at exploring the implementation of the SHA-256 cryptographic hashing algorithm using quantum computing techniques. Leveraging Qiskit, IBM's quantum computing framework, this project endeavors to translate the SHA-256 operations into quantum circuits, exploring the potential impact of quantum computing on cryptographic hashing.

## Objectives
- **Advance Cryptographic Techniques**: Investigate how quantum computing can be integrated into traditional cryptographic methods.
- **Educational Resource**: Serve as a learning tool for enthusiasts and professionals interested in the intersection of quantum computing and cryptography.
- **Research Platform**: Provide a base for experimental development, testing, and discussion of quantum cryptographic methods.

## Features
- **Quantum Circuit Implementation**: Implementation of SHA-256 as a quantum circuit using Qiskit.
- **Simulation Tools**: Utilize Aer simulator from Qiskit for local testing and simulations.

## Installation

### Prerequisites
- Python 3.8+
- [Qiskit](https://qiskit.org)

### Setup
Clone the repository and install the required packages.

```bash
git clone https://github.com/oracleagent/qSHA256.git
cd qSHA256
pip install -r requirements.txt
```

## Usage
Navigate to the project directory and run the Python script to execute the quantum SHA-256 circuit.

```python
python 0.py
```

This will initiate the quantum circuit and output the results of the hash operation simulated on a quantum computer.
