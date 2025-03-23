import asyncio
from dsl_runtime import QuantumBeastRuntime
import matplotlib.pyplot as plt
from typing import List, Dict
import time

async def visualize_beast_state(beast_id: str, runtime: QuantumBeastRuntime, history: List[Dict]):
    """Visualizes the quantum beast state"""
    plt.figure(figsize=(15, 10))
    
    # Plot entropy over time
    plt.subplot(2, 2, 1)
    entropies = [h['entropy'] for h in history]
    plt.plot(entropies)
    plt.title('Entropy Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    
    # Plot coherence over time
    plt.subplot(2, 2, 2)
    coherences = [h['coherence'] for h in history]
    plt.plot(coherences)
    plt.title('Coherence Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Coherence')
    
    # Plot entanglement matrix
    plt.subplot(2, 2, 3)
    if runtime.entanglement_matrix is not None:
        plt.imshow(runtime.entanglement_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('Entanglement Matrix')
    
    # Plot quantum state
    plt.subplot(2, 2, 4)
    beast = runtime.beast_states[beast_id]
    statevector = beast['state'].statevector
    plt.plot(np.abs(statevector.data))
    plt.title('Quantum State')
    plt.xlabel('State Index')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(f'beast_state_{int(time.time())}.png')
    plt.close()

async def main():
    # Initialize runtime
    runtime = QuantumBeastRuntime()
    
    # Read beast program
    with open('examples/genesis.beast', 'r') as f:
        source = f.read()
    
    # Spawn initial beast
    beast_id = await runtime.spawn_beast("ATGCATGCATGCATGCATGCATGCATGCATGC")
    print(f"Spawned beast with ID: {beast_id}")
    
    # Initialize history
    history = []
    
    try:
        # Run evolution cycle
        for i in range(100):
            # Evolve beast
            state = await runtime.evolve_beast(beast_id, "input_data")
            history.append(state)
            
            # Sense oracle influences
            weather = await runtime.sense_oracle("weather")
            news = await runtime.sense_oracle("news")
            
            # Emit hash
            hash_value = await runtime.emit_hash(beast_id)
            
            # Recall memory
            memory = await runtime.recall_memory(beast_id, "query")
            
            # Visualize state every 10 iterations
            if i % 10 == 0:
                await visualize_beast_state(beast_id, runtime, history)
                print(f"Iteration {i}:")
                print(f"  Entropy: {state['entropy']:.4f}")
                print(f"  Coherence: {state['coherence']:.4f}")
                print(f"  Hash: {hash_value[:8]}...")
                print(f"  Weather influence: {weather['weather']:.4f}")
                print(f"  News influence: {news['news']:.4f}")
                print()
            
            # Wait for next iteration
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nEvolution cycle interrupted")
    finally:
        # Final visualization
        await visualize_beast_state(beast_id, runtime, history)
        print("\nEvolution complete")
        print(f"Final entropy: {history[-1]['entropy']:.4f}")
        print(f"Final coherence: {history[-1]['coherence']:.4f}")

if __name__ == "__main__":
    asyncio.run(main()) 
