#!/usr/bin/env python3
"""
Quantum SHA-256 Demo

This script demonstrates the quantum SHA-256 implementation by:
1. Creating a quantum circuit with SHA-256 components
2. Running the circuit on a simulator
3. Analyzing the quantum state
4. Generating visualizations of entropy and coherence

This is for educational purposes only.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Import our quantum SHA-256 implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsha256 import QuantumSHA256, calculate_entropy, calculate_coherence
from qsha256.utils import analyze_state_evolution
from qiskit.quantum_info import Statevector


def create_sample_message() -> List[int]:
    """Create a sample message for demonstration."""
    # Convert "Hello, Quantum!" to binary
    message = "Hello, Quantum!"
    message_bits = []
    for char in message:
        # Convert character to 8-bit binary
        char_bits = format(ord(char), '08b')
        message_bits.extend([int(bit) for bit in char_bits])
    
    # Pad to a reasonable length for demonstration
    while len(message_bits) < 256:
        message_bits.append(0)
    
    return message_bits[:256]  # Take first 256 bits


def run_quantum_demo():
    """Run the main quantum SHA-256 demonstration."""
    print("=== Quantum SHA-256 Demo ===\n")
    
    # Create sample message
    message_bits = create_sample_message()
    print(f"Sample message: 'Hello, Quantum!'")
    print(f"Message bits (first 32): {message_bits[:32]}")
    print(f"Total message length: {len(message_bits)} bits\n")
    
    # Initialize quantum SHA-256
    print("Creating quantum SHA-256 circuit...")
    qsha256 = QuantumSHA256(num_qubits=8, enable_error_correction=True)
    
    # Create the quantum circuit
    circuit = qsha256.create_circuit(message_bits)
    print(f"Circuit created with {circuit.num_qubits} qubits")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Circuit size: {circuit.size()}")
    print(f"Gate counts: {circuit.count_ops()}\n")
    
    # Analyze the quantum state
    print("Analyzing quantum state...")
    state_analysis = qsha256.get_state_analysis()
    
    if 'error' in state_analysis:
        print(f"State analysis error: {state_analysis['error']}")
        print("Using basic circuit metrics instead...")
        print(f"Circuit depth: {state_analysis.get('circuit_depth', 'N/A')}")
        print(f"Circuit size: {state_analysis.get('circuit_size', 'N/A')}")
        print(f"Number of qubits: {state_analysis.get('num_qubits', 'N/A')}")
    else:
        print(f"Quantum entropy: {state_analysis.get('entropy', 'N/A'):.4f}")
        print(f"Quantum coherence: {state_analysis.get('coherence', 'N/A'):.4f}")
        print(f"State purity: {state_analysis.get('purity', 'N/A'):.4f}")
        print(f"Entanglement entropy: {state_analysis.get('entanglement_entropy', 'N/A'):.4f}")
        print(f"Quantum volume: {state_analysis.get('quantum_volume', 'N/A'):.4f}")
    
    print()
    
    # Simulate the circuit
    print("Simulating quantum circuit...")
    simulation_results = qsha256.simulate(shots=1024)
    
    if 'error' in simulation_results:
        print(f"Simulation error: {simulation_results['error']}")
    else:
        print(f"Simulation completed with {simulation_results['shots']} shots")
        if simulation_results['most_frequent']:
            print(f"Most frequent measurement: {simulation_results['most_frequent'][:16]}...")
            print(f"Number of unique outcomes: {len(simulation_results['counts'])}")
    
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    create_visualizations(state_analysis, simulation_results)
    
    print("Demo completed! Check the generated plots.")
    return state_analysis, simulation_results


def create_visualizations(state_analysis: dict, simulation_results: dict):
    """Create visualizations of the quantum state analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quantum SHA-256 Analysis', fontsize=16)
    
    # Plot 1: Quantum metrics
    ax1 = axes[0, 0]
    if 'error' not in state_analysis:
        metrics = ['entropy', 'coherence', 'purity', 'entanglement_entropy']
        values = [state_analysis.get(metric, 0) for metric in metrics]
        bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Quantum State Metrics')
        ax1.set_ylabel('Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'State analysis\nfailed', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Quantum State Metrics (Error)')
    
    # Plot 2: Circuit metrics
    ax2 = axes[0, 1]
    if 'error' not in state_analysis:
        circuit_metrics = ['depth', 'size', 'num_qubits']
        circuit_values = [state_analysis.get(metric, 0) for metric in circuit_metrics]
        bars = ax2.bar(circuit_metrics, circuit_values, color=['#9467bd', '#8c564b', '#e377c2'])
        ax2.set_title('Circuit Metrics')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, circuit_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(circuit_values) * 0.01,
                    f'{value}', ha='center', va='bottom')
    else:
        # Show basic circuit info if available
        basic_info = [state_analysis.get('circuit_depth', 0), 
                     state_analysis.get('circuit_size', 0),
                     state_analysis.get('num_qubits', 0)]
        if any(basic_info):
            ax2.bar(['depth', 'size', 'qubits'], basic_info, color=['#9467bd', '#8c564b', '#e377c2'])
            ax2.set_title('Basic Circuit Metrics')
        else:
            ax2.text(0.5, 0.5, 'Circuit metrics\nunavailable', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Circuit Metrics (Unavailable)')
    
    # Plot 3: Simulation results
    ax3 = axes[1, 0]
    if 'error' not in simulation_results and simulation_results.get('counts'):
        # Show distribution of measurement outcomes
        counts = simulation_results['counts']
        outcomes = list(counts.keys())[:20]  # Show first 20 outcomes
        frequencies = [counts[outcome] for outcome in outcomes]
        
        bars = ax3.bar(range(len(outcomes)), frequencies, color='#17becf')
        ax3.set_title('Measurement Outcomes (Top 20)')
        ax3.set_xlabel('Outcome Index')
        ax3.set_ylabel('Frequency')
        
        # Add frequency labels
        for i, freq in enumerate(frequencies):
            ax3.text(i, freq + max(frequencies) * 0.01, str(freq), 
                    ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Simulation\nfailed', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Measurement Outcomes (Error)')
    
    # Plot 4: Quantum volume comparison
    ax4 = axes[1, 1]
    if 'error' not in state_analysis:
        # Create a simple quantum volume visualization
        quantum_volume = state_analysis.get('quantum_volume', 0)
        depth = state_analysis.get('depth', 0)
        num_qubits = state_analysis.get('num_qubits', 0)
        
        # Show quantum volume as a heatmap-like visualization
        volume_data = np.array([[quantum_volume]])
        im = ax4.imshow(volume_data, cmap='viridis', aspect='auto')
        ax4.set_title(f'Quantum Volume: {quantum_volume:.2f}')
        ax4.set_xlabel('Circuit Complexity')
        ax4.set_ylabel('Qubit Count')
        
        # Add text annotation
        ax4.text(0, 0, f'Depth: {depth}\nQubits: {num_qubits}', 
                ha='center', va='center', color='white', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Quantum volume\nunavailable', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Quantum Volume (Unavailable)')
    
    plt.tight_layout()
    plt.savefig('quantum_sha256_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'quantum_sha256_analysis.png'")
    
    # Create a simple circuit diagram if possible
    try:
        from qiskit.visualization import circuit_drawer
        qsha256 = QuantumSHA256(num_qubits=64)  # Smaller circuit for visualization
        simple_circuit = qsha256.create_circuit([0] * 64)
        
        # Draw a simplified version
        circuit_fig = circuit_drawer(simple_circuit, output='mpl', 
                                   style={'backgroundcolor': 'white'},
                                   scale=0.8)
        circuit_fig.savefig('quantum_sha256_circuit.png', dpi=300, bbox_inches='tight')
        print("Circuit diagram saved as 'quantum_sha256_circuit.png'")
        
    except Exception as e:
        print(f"Could not generate circuit diagram: {e}")


def main():
    """Main function to run the demo."""
    try:
        state_analysis, simulation_results = run_quantum_demo()
        
        print("\n=== Summary ===")
        if 'error' not in state_analysis:
            print("✓ Quantum circuit created successfully")
            print("✓ State analysis completed")
            print("✓ Simulation executed")
            print("✓ Visualizations generated")
        else:
            print("⚠ Some components failed, but basic functionality demonstrated")
        
        print("\nThis demo shows how SHA-256 components can be implemented")
        print("using quantum circuits. The implementation includes:")
        print("- Quantum XOR, AND, Ch, and Maj functions")
        print("- Sigma functions for bit rotation")
        print("- Error correction and parallelism features")
        print("- State analysis and visualization")
        print("\nNote: This is for educational purposes only!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install qiskit numpy matplotlib")


if __name__ == "__main__":
    main()
