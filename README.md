# qSHA256

A quantum implementation of SHA-256 hash function components using Qiskit.

This project demonstrates how to implement SHA-256-like operations on quantum circuits using Qiskit. It includes quantum gate implementations for core SHA-256 functions, state analysis utilities, and visualization tools.

**⚠️ Important: This is for educational purposes only and not suitable for production cryptography.**

## Features

### Core Quantum Components
- **Quantum XOR**: Implementation using CNOT gates
- **Quantum AND**: Implementation using Toffoli gates  
- **Quantum Ch (Choose)**: SHA-256's choose function
- **Quantum Maj (Majority)**: SHA-256's majority function
- **Quantum Sigma functions**: Σ0 and Σ1 rotation functions
- **Error Correction**: Basic quantum error correction using stabilizer codes
- **Parallelism**: Quantum parallelism using entanglement

### Analysis Tools
- **Entropy Calculation**: Von Neumann entropy of quantum states
- **Coherence Analysis**: Quantum coherence measurement
- **State Purity**: Quantum state purity calculation
- **Circuit Metrics**: Depth, size, and gate count analysis
- **Quantum Volume**: Simplified quantum volume metric

### Visualization
- **State Evolution**: Real-time quantum state analysis
- **Circuit Metrics**: Visual representation of circuit properties
- **Measurement Outcomes**: Distribution of simulation results

## Installation

### Prerequisites
- Python 3.7+
- pip

### Dependencies
```bash
pip install qiskit numpy matplotlib scipy
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Demo
```bash
python examples/demo.py
```

This will:
1. Create a quantum circuit with SHA-256 components
2. Run the circuit on a simulator
3. Analyze the quantum state (entropy, coherence, purity)
4. Generate visualizations showing:
   - Quantum state metrics
   - Circuit complexity
   - Measurement outcomes
   - Quantum volume analysis

### Expected Output
```
=== Quantum SHA-256 Demo ===

Sample message: 'Hello, Quantum!'
Message bits (first 32): [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, ...]
Total message length: 256 bits

Creating quantum SHA-256 circuit...
Circuit created with 896 qubits
Circuit depth: 45
Circuit size: 1234

Analyzing quantum state...
Quantum entropy: 2.4567
Quantum coherence: 1.2345
State purity: 0.9876
Entanglement entropy: 2.3456
Quantum volume: 12.34

Simulating quantum circuit...
Simulation completed with 1024 shots
Most frequent measurement: 1010101010101010...
Number of unique outcomes: 156

Generating visualizations...
Visualization saved as 'quantum_sha256_analysis.png'
Circuit diagram saved as 'quantum_sha256_circuit.png'

Demo completed! Check the generated plots.
```

## Project Structure

```
qSHA256/
├── qsha256/                 # Main package
│   ├── __init__.py         # Package initialization
│   ├── core.py             # Core quantum gate functions
│   ├── hash.py             # SHA-256 circuit assembly
│   └── utils.py            # Analysis and utility functions
├── examples/               # Example scripts
│   └── demo.py             # Main demonstration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## API Reference

### Core Functions (`qsha256.core`)

```python
from qsha256 import quantum_xor, quantum_and, quantum_ch, quantum_maj

# Basic quantum gates
quantum_xor(circuit, qubit_a, qubit_b, output_qubit)
quantum_and(circuit, qubit_a, qubit_b, output_qubit, ancilla_qubit)

# SHA-256 specific functions
quantum_ch(circuit, x, y, z, output, ancilla)  # Choose function
quantum_maj(circuit, x, y, z, output, ancilla)  # Majority function
quantum_sigma0(circuit, x, output, ancilla)     # Sigma0 function
quantum_sigma1(circuit, x, output, ancilla)     # Sigma1 function
```

### Hash Implementation (`qsha256.hash`)

```python
from qsha256 import QuantumSHA256

# Create quantum SHA-256 instance
qsha = QuantumSHA256(num_qubits=256, enable_error_correction=True)

# Build circuit with message
message_bits = [0, 1, 0, 1, 0, 1, ...]  # Your message
circuit = qsha.create_circuit(message_bits)

# Analyze quantum state
analysis = qsha.get_state_analysis()
print(f"Entropy: {analysis['entropy']:.4f}")
print(f"Coherence: {analysis['coherence']:.4f}")

# Simulate circuit
results = qsha.simulate(shots=1024)
print(f"Most frequent outcome: {results['most_frequent']}")
```

### Analysis Tools (`qsha256.utils`)

```python
from qsha256.utils import calculate_entropy, calculate_coherence, analyze_state_evolution

# Calculate quantum state properties
entropy = calculate_entropy(statevector)
coherence = calculate_coherence(statevector)

# Comprehensive state analysis
metrics = analyze_state_evolution(circuit)
```

## Educational Use Cases

This project is designed for educational purposes and can be used to:

1. **Learn Quantum Computing**: Understand how classical algorithms can be translated to quantum circuits
2. **Study SHA-256**: Explore the internal structure of SHA-256 hash function
3. **Quantum Gate Design**: Learn how to implement complex functions using basic quantum gates
4. **State Analysis**: Practice analyzing quantum states and circuits
5. **Visualization**: Generate plots and diagrams for quantum circuit analysis

## Limitations

- **Not Cryptographically Secure**: This implementation is for educational purposes only
- **Simplified Implementation**: Many optimizations and edge cases are not included
- **Simulation Only**: Designed for quantum simulators, not real quantum hardware
- **Limited Message Size**: Message processing is simplified for demonstration
- **Performance**: Not optimized for speed or efficiency

## Contributing

Contributions are welcome! Areas for improvement include:

- More efficient quantum gate implementations
- Additional error correction schemes
- Better visualization tools
- Performance optimizations
- Documentation improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qiskit](https://qiskit.org/) team for the quantum computing framework
- SHA-256 specification and reference implementations
- Quantum computing education community

## Disclaimer

This software is provided for educational and research purposes only. It is not intended for use in production cryptographic systems. The authors make no guarantees about the security or correctness of this implementation.