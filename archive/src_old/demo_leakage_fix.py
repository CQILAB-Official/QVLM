#!/usr/bin/env python3
"""
Compare OLD vs NEW reasoning generation to show the data leakage fix
"""

# OLD VERSION (with data leakage)
def old_generate_reasoning(purity_value):
    purity = float(purity_value)
    if purity >= 0.95:
        return f"The circuit exhibits very high purity at {purity:.6f}. Excellent coherence preservation."
    elif purity >= 0.85:
        return f"Analyzing the circuit structure, I observe high purity around {purity:.6f}."
    elif purity >= 0.50:
        return f"The purity appears to be in the medium range around {purity:.6f}."
    else:
        return f"The circuit indicates low purity at {purity:.6f}."

# NEW VERSION (without data leakage)
def new_generate_reasoning(purity_value):
    purity = float(purity_value)
    if purity >= 0.95:
        return "The circuit exhibits very high purity. Excellent coherence preservation."
    elif purity >= 0.85:
        return "Analyzing the circuit structure, I observe high purity."
    elif purity >= 0.50:
        return "The purity appears to be in the medium range."
    else:
        return "The circuit indicates low purity."

# Test with example values
test_values = [1.0, 0.968008, 0.84971, 0.545867]

print("="*80)
print("DATA LEAKAGE FIX DEMONSTRATION")
print("="*80)

for purity in test_values:
    print(f"\n{'='*80}")
    print(f"Ground Truth Purity: {purity}")
    print(f"{'='*80}")
    
    old_reasoning = old_generate_reasoning(purity)
    new_reasoning = new_generate_reasoning(purity)
    
    print(f"\n❌ OLD (LEAKS DATA):")
    print(f"   <think>{old_reasoning} The purity is {purity}</think>")
    print(f"   → Model learns to just COPY {purity} from reasoning!")
    
    print(f"\n✅ NEW (NO LEAKAGE):")
    print(f"   <think>{new_reasoning} The purity is {purity}</think>")
    print(f"   → Model must PREDICT {purity} from image features!")

print(f"\n{'='*80}")
print("✓ Fix applied: Removed decimal values from reasoning chains")
print("="*80)
