#!/usr/bin/env python3
"""
Test script to verify chain-of-thought generation for purity predictions
"""

def generate_purity_reasoning(purity_value, entangled_status="Yes"):
    """
    Generate synthetic chain-of-thought reasoning for purity predictions.
    This helps the model learn to associate visual features with decimal values.
    
    Args:
        purity_value: The ground truth purity (0.0 to 1.0)
        entangled_status: Whether the circuit is entangled ("Yes" or "No")
    
    Returns:
        A reasoning string that explains the purity assessment
    """
    try:
        purity = float(purity_value)
    except (ValueError, TypeError):
        # If we can't parse, return minimal reasoning
        return "Unable to determine purity from the circuit visualization."
    
    # Generate reasoning based on purity ranges
    if purity >= 0.99:
        reasoning = (
            "The quantum circuit diagram shows a pure quantum state. "
            "The gate structure appears to preserve coherence throughout, "
            "with minimal decoherence or mixing effects visible. "
            "The state evolution maintains near-perfect purity."
        )
    elif purity >= 0.95:
        reasoning = (
            f"The circuit exhibits very high purity at {purity:.6f}. "
            "The gate operations show excellent coherence preservation, "
            "with only minimal environmental coupling or decoherence. "
            "The state is very close to pure but shows slight mixing."
        )
    elif purity >= 0.85:
        reasoning = (
            f"Analyzing the circuit structure, I observe high purity around {purity:.6f}. "
            "While the quantum operations maintain good coherence, "
            "there are some signs of decoherence or partial tracing effects. "
            "The state is predominantly pure with moderate mixing."
        )
    elif purity >= 0.70:
        reasoning = (
            f"The circuit demonstrates moderate purity of approximately {purity:.6f}. "
            "The gate sequence shows noticeable decoherence effects, "
            "suggesting interaction with the environment or partial measurements. "
            "The state exhibits significant but not dominant mixing."
        )
    elif purity >= 0.50:
        reasoning = (
            f"The purity appears to be in the medium range around {purity:.6f}. "
            "The circuit shows substantial mixing, possibly from strong environmental coupling "
            "or measurement-induced decoherence. The pure and mixed components are comparable."
        )
    else:
        reasoning = (
            f"The circuit indicates low purity at {purity:.6f}. "
            "Strong decoherence and mixing effects are evident, "
            "suggesting significant environmental interaction or thermal effects. "
            "The state is heavily mixed."
        )
    
    # Add entanglement context if applicable
    if entangled_status == "Yes" and purity < 1.0:
        reasoning += " The presence of entanglement combined with sub-unity purity suggests a mixed entangled state."
    elif entangled_status == "No" and purity == 1.0:
        reasoning += " The separable structure with perfect purity indicates a pure product state."
    
    return reasoning


if __name__ == "__main__":
    # Test with sample values from the dataset
    test_cases = [
        (1.0, "No"),
        (0.962695, "Yes"),
        (0.968008, "Yes"),
        (0.782576, "Yes"),
        (0.533982, "Yes"),
    ]
    
    print("=" * 80)
    print("Chain-of-Thought Generation Examples")
    print("=" * 80)
    
    for purity, entangled in test_cases:
        reasoning = generate_purity_reasoning(purity, entangled)
        print(f"\nPurity: {purity} | Entangled: {entangled}")
        print(f"Reasoning: {reasoning}")
        print("-" * 80)
    
    print("\n✓ Chain-of-thought generation test complete!")
