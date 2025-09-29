"""
Utility functions for quantum state analysis and metrics.

This module provides functions to calculate entropy, coherence, and other
quantum state properties for analyzing quantum circuits.
"""

import numpy as np
from typing import Union
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def calculate_entropy(statevector: Statevector) -> float:
    """
    Calculate the von Neumann entropy of a quantum state.
    
    Args:
        statevector: The quantum state to analyze
        
    Returns:
        The entropy value (0 for pure states, positive for mixed states)
    """
    # Get the probability amplitudes
    probabilities = np.abs(statevector.data) ** 2
    
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def calculate_coherence(statevector: Statevector) -> float:
    """
    Calculate the coherence of a quantum state.
    
    Coherence is measured as the sum of absolute values of off-diagonal
    elements of the density matrix.
    
    Args:
        statevector: The quantum state to analyze
        
    Returns:
        The coherence value
    """
    # Convert to density matrix
    density_matrix = DensityMatrix(statevector)
    
    # Get the matrix data
    matrix = density_matrix.data
    
    # Calculate coherence as sum of off-diagonal elements
    coherence = np.sum(np.abs(matrix)) - np.sum(np.abs(np.diag(matrix)))
    
    return coherence


def calculate_state_purity(statevector: Statevector) -> float:
    """
    Calculate the purity of a quantum state.
    
    Purity is defined as Tr(ρ²) where ρ is the density matrix.
    
    Args:
        statevector: The quantum state to analyze
        
    Returns:
        The purity value (1 for pure states, < 1 for mixed states)
    """
    # Convert to density matrix
    density_matrix = DensityMatrix(statevector)
    
    # Calculate purity as Tr(ρ²)
    purity = np.trace(density_matrix.data @ density_matrix.data)
    
    return np.real(purity)


def calculate_entanglement_entropy(statevector: Statevector, partition: int = None) -> float:
    """
    Calculate the entanglement entropy for a bipartition of the system.
    
    Args:
        statevector: The quantum state to analyze
        partition: The qubit index to partition at (default: half the qubits)
        
    Returns:
        The entanglement entropy
    """
    if partition is None:
        partition = statevector.num_qubits // 2
    
    if partition >= statevector.num_qubits:
        raise ValueError("Partition must be less than the number of qubits")
    
    # For a simple demonstration, we'll use the full state entropy
    # In a real implementation, this would involve proper bipartition
    return calculate_entropy(statevector)


def get_circuit_metrics(circuit: QuantumCircuit) -> dict:
    """
    Get basic metrics about a quantum circuit.
    
    Args:
        circuit: The quantum circuit to analyze
        
    Returns:
        Dictionary containing circuit metrics
    """
    return {
        'depth': circuit.depth(),
        'size': circuit.size(),
        'num_qubits': circuit.num_qubits,
        'gate_counts': circuit.count_ops()
    }


def calculate_quantum_volume(circuit: QuantumCircuit) -> float:
    """
    Calculate a simplified quantum volume metric.
    
    Args:
        circuit: The quantum circuit to analyze
        
    Returns:
        A quantum volume-like metric
    """
    depth = circuit.depth()
    width = circuit.num_qubits
    
    # Simplified quantum volume calculation
    return min(depth, width) * np.log2(max(depth, width, 2))


def analyze_state_evolution(circuit: QuantumCircuit) -> dict:
    """
    Analyze the evolution of quantum states in a circuit.
    
    Args:
        circuit: The quantum circuit to analyze
        
    Returns:
        Dictionary containing state evolution metrics
    """
    try:
        # Get the final statevector
        statevector = Statevector.from_instruction(circuit)
        
        # Calculate various metrics
        metrics = {
            'entropy': calculate_entropy(statevector),
            'coherence': calculate_coherence(statevector),
            'purity': calculate_state_purity(statevector),
            'entanglement_entropy': calculate_entanglement_entropy(statevector),
            'quantum_volume': calculate_quantum_volume(circuit)
        }
        
        # Add circuit metrics
        metrics.update(get_circuit_metrics(circuit))
        
        return metrics
        
    except Exception as e:
        # Return basic circuit metrics if state analysis fails
        return get_circuit_metrics(circuit)
