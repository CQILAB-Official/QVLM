from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = """
    DO NOT SUMMARIZE. ONLY OUTPUT RAW TAGGED BLOCKS.
    Generate random QPE circuit with depth 4 and number of qubits 6 (Qiskit qasm), using logical gate
Supposed that the output is image, you need to make chain of thought on how to reverse the image into thinking process so it can summarize the details, the structures, and classify number of qubits along with registers, also how to classify it as grover/qpe/qml etc

This is the example format:

<think>  
1. Based on the image, consist of pattern .. there is CCNOT and X gate as oracle then it is classified as Grover (as detail as possible) (Might be different for QPE, QML etc) 
(You may add more steps here as detail as possible)

2. Number of qubits, how many quantum classical registers, is there measurement, reverse engineering from image to thinking process, how to classify number of qubits, registers, and measurement

3. Composition - Logical GATE (reverse OCR):  
[Q0] --- [X] --- [H] --- [Cnot1]  
[Q1] --- [X] --- [H] --- [Cnot1]  

</think>  

<OPENQASM code>  
 ...  
 ...  
</OPENQASM code>  
"""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
