import os
import base64
from openai import OpenAI
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

# Output configuration
OUTPUT_FILE = "inference_results-wigner-chatgpt41.csv"

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string for OpenAI API.
    Strips all metadata including filename, EXIF data, etc. - only sends pure pixel data.
    """
    from io import BytesIO
    
    # Load image with PIL
    img = Image.open(image_path)
    
    # Create a new image from pixel data only (strips all metadata)
    # This ensures no filename, EXIF, or other metadata is included
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(list(img.getdata()))
    
    # Save to BytesIO buffer as PNG (lossless, no metadata)
    buffer = BytesIO()
    clean_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_inference_chatgpt41(image_path: str, user_input: str, temperature: float = 0.3, 
                            max_tokens: int = 500, api_key: str = None) -> str:
    """
    Run inference using OpenAI's GPT-4.1 API with vision capabilities.
    
    Args:
        image_path: Path to the image file
        user_input: The prompt text
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
    
    Returns:
        Generated text response from GPT-4.1
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY"))
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Create the message with image and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        }
    ]
    
    # Call GPT-4.1 API
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Extract generated text
    generated_text = response.choices[0].message.content
    
    return generated_text

if __name__ == "__main__":
    CSV_FILENAME = 'Dataset/wigner_refactor.csv' 
    data = pd.read_csv(CSV_FILENAME)
    
    required_columns = ['image', 'ground_truth']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the CSV file.")
    
    # Shuffle the dataset with a controlled random state
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data into train and test sets
    _, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Drop rows with NaN and reset indices
    test_data = test_data.reset_index(drop=True)
    
    BEST_PROMPT = (
        "You are given a grayscale image representing a quantum optical state. "
        "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, random state, etc.) "
        "as well as its key parameters (alpha/number of photons/density, truncated Hilbert space dimension, and the linear space range). "
        "Please provide your answer in the following format: "
        "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] with [KEY parameters] equal to [VALUE] represented using a truncated Hilbert space of dimension d = [DIM_VALUE], "
        "in the linear space from -[LINVALUE] to [LINVALUE]. \""
        "Under a compact binary encoding, this truncated space could be represented using ⌈log2([DIM_VALUE])⌉ = [TOTAL_QUBIT] qubits."
        "Then extract your reasoning on how you determined the state, parameters, and number of qubits from the image, "
        "YOU MUST PROVIDE <think></think> BRACKET FOR THE THINKING PROCESS, "
        "DON'T TAKE THE THINKING OR CONCLUSION FROM THE IMAGE FORMAT OR IMAGE NAME OR IMAGE METADATA, "
        "YOU SHOULD ANSWER IN THIS EXACT FORMAT IN THE AFTER THE THINKING PROCESS:"
        "This is a [STATE TYPE] with [KEY parameters] equal to [VALUE] represented using a truncated Hilbert space of dimension d = [DIM_VALUE], "
        "in the linear space from -[LINVALUE] to [LINVALUE]. "
        "Under a compact binary encoding, this truncated space could be represented using ⌈log2([DIM_VALUE])⌉ = [TOTAL_QUBIT] qubits."
    )

    # Prepare test data (limit to 5 samples for quick testing)
    # NUM_TEST_SAMPLES = 5
    x_test = test_data['image'][:]
    images = test_data['image'][:]
    y_test = test_data['ground_truth'][:]

    output_file = OUTPUT_FILE

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["test_case", "image", "generated", "ground_truth"])

        for i in range(len(x_test)):
            # Prepend Dataset/ to image path since CSV paths are relative to Dataset/
            image_path = os.path.join("Dataset", images[i])
            
            # Verify image exists
            if not os.path.exists(image_path):
                print(f"WARNING: Image not found: {image_path}. Skipping...")
                continue
            
            # Construct the full prompt
            user_input = BEST_PROMPT + x_test[i]
            
            try:
                generated = run_inference_chatgpt41(
                    image_path,
                    user_input,
                    temperature=0.3,
                    max_tokens=500,
                )
            except Exception as e:
                print(f"Error processing row {i + 1}: {str(e)}")
                generated = f"ERROR: {str(e)}"

            # Write row directly
            writer.writerow([
                i + 1,
                images[i],
                generated,
                y_test[i],
            ])

            print(f"Saved row {i + 1} / {len(x_test)} to {output_file}")

    print(f"Saved {len(x_test)} rows to {output_file}")
