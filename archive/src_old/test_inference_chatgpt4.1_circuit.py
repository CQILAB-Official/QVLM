import os
import base64
from openai import OpenAI
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Output configuration
OUTPUT_FILE = "inference_results-circuit-chatgpt41.csv"

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

def process_csv(csv_path: str, max_io: int, seed=42, test_frac=0.20):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    _, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    test_df = test_df.reset_index(drop=True)

    out_file = "ResultUpdate/" + os.path.splitext(csv_path)[0] + "_chatgpt41_inference-v2.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    header = ["image"]
    for i in range(1, max_io + 1):
        header.extend([f"input_{i}", f"prediction_{i}", f"output_{i}"])

    start_idx = 0
    file_mode = "w"
    
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file)
            start_idx = len(existing_df)
            if start_idx > 0:
                file_mode = "a"
                print(f"Found existing output file. Resuming from index {start_idx} (row {start_idx + 1})...")
        except Exception as e:
            print(f"Error reading existing file: {e}. Starting from scratch.")

    with open(out_file, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if file_mode == "w":
            writer.writerow(header)

        for idx, (_, row) in enumerate(test_df.iterrows()):
            if idx < start_idx:
                continue

            print(f"[{csv_path}] Processing row {idx+1}/{len(test_df)}", flush=True)
            img_path = row["image"]
            
            if not os.path.exists(img_path):
                print(f"WARNING: Image not found: {img_path}. Skipping...")
                row_out = [img_path]
                for n in range(1, max_io + 1):
                    row_out.extend(["", "ERROR: Image not found", ""])
                writer.writerow(row_out)
                continue
                
            row_out = [img_path]

            for n in range(1, max_io + 1):
                input_col, out_col = f"input_{n}", f"output_{n}"
                if input_col not in row or pd.isna(row[input_col]):
                    row_out.extend(["", "", ""])
                    continue

                user_prompt = str(row[input_col])
                
                try:
                    pred = run_inference_chatgpt41(
                        image_path=img_path,
                        user_input=user_prompt,
                        temperature=0.3,
                        max_tokens=1500,
                    )
                except Exception as e:
                    print(f"Error processing row {idx + 1}, input {n}: {str(e)}")
                    pred = f"ERROR: {str(e)}"

                row_out.extend([row[input_col], pred, row[out_col]])
                print(f"Finish processing row {idx+1}/{len(test_df)}, input {n}/{max_io}")

            writer.writerow(row_out)

    print(f"Wrote {len(test_df)} rows to {out_file}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
        # For testing purposes we'll allow it to continue in case it's handled differently or set inside the IDE

    tasks = [
        # ("case_study_entanglement.csv", 2),
        ("case_study_circuit_patched.csv", 7),
    ]
    for path, max_cols in tasks:
        process_csv(path, max_cols)
