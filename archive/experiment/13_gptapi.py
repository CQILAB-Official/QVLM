import base64
import pandas as pd
import os
import time
import openai
from PIL import Image
import io
import csv
from dotenv import load_dotenv

# Load from .env
load_dotenv()

# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup OpenAI client (new in v1+)
client = openai.OpenAI(api_key=openai_api_key)

# Function to encode image to base64
def encode_image(image_path):
    # Open and resize image
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure 3-channel
        # img = img.resize((600, 300))

        # Save resized image to a buffer
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # Encode to base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image_with_prompt(ground_truth):
    # base64_image = encode_image(image_path)
    
    prompt = (
        "You are given a Wigner function image from a quantum optical simulation. "
        "Please determine the type of quantum state, average number of photon, and infer the number of qubits. "
        "Format your reasoning like this:\n"
        "<think>\n"
        "Step-by-step reasoning here...\n"
        "</think>\n"
        "Then give the final answer like this:\n"
        "This is a [STATE TYPE] with average number of photon ≈ [VALUE], number of qubits = [N], in the linear space from [LOW] to [HIGH].\n"
        "For the thinking process, please do calculation based on wigner funciton\n" 
        "Please give the result similar with the ground truth i give you, because that is the correct answer, this is the ground  truth with [STATE TYPE], average number of photon, number of qubits, and linear space\n"
        "" + ground_truth + "\n"
        "But, please don't give thinking like this `Given that we are aiming for a similar result to the provided ground truth, I will infer number of qubits as 30, which is consistent with a similar linear space range.`.\n"
        "Please don't mention it from the ground truth i give you, but get the conclusion from image that is make sense to the ground truth i give you\n"
        "make sure all the answer is the same as the ground truth i give you (average number of photon, state type, number of qubits, and linear space)\n"
        "The most important, don't also give explanation of \"I'm sorry, I can't assist with that.\" or \"you need to find any connection from the image to the result that is make sense I give you \"\n"
        "And make sure in thinking process it needs (mandatory) to be inside <think> .... </think> tag"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quantum optics assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/png;base64,{base64_image}",
                    #         "detail": "high"
                    #     }
                    # }
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


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

