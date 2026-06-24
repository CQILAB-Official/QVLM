import torch
import base64
from io import BytesIO
from PIL import Image

class EndpointHandler():
    def __init__(self, path=""):
        """
        This method runs once when your Hugging Face space/endpoint starts.
        You should load your router model and expert models here.
        """
        print("Initializing MoE System...")
        
        # 1. Load the Router Model
        # Example using a small, fast VLM as the router:
        # self.router_model = load_model("Qwen/Qwen2.5-VL-3B-Instruct")
        
        # 2. Load the Expert Models 
        # (Be mindful of VRAM. If experts are too large, you might need to 
        # use API calls to other endpoints instead of loading them locally).
        # self.math_expert = load_model("...") 
        # self.ocr_expert = load_model("...")
        
        print("All models successfully loaded into memory!")

    def __call__(self, data):
        """
        This method is called every time an API request is made.
        Expected input format:
        {
            "inputs": {
                "image": "<base64_encoded_string>",
                "prompt": "User's question goes here"
            }
        }
        """
        # Parse inputs
        inputs = data.pop("inputs", data)
        image_b64 = inputs.get("image", None)
        prompt = inputs.get("prompt", "")

        if not image_b64 or not prompt:
            return {"error": "Both 'image' (base64) and 'prompt' must be provided."}
            
        # Decode image
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}

        # --- STEP 1: ROUTING ---
        # The router analyzes the image and prompt to decide which expert to use.
        route_decision = self._route_request(image, prompt)
        print(f"Router selected expert: {route_decision}")

        # --- STEP 2: PROCESSING VIA EXPERT ---
        # Send the original prompt and image to the selected expert model.
        if route_decision == "MATH_EXPERT":
            result = self._call_math_expert(image, prompt)
        elif route_decision == "OCR_EXPERT":
            result = self._call_ocr_expert(image, prompt)
        else:
            result = self._call_general_expert(image, prompt)

        # --- STEP 3: RETURN FINAL RESULT ---
        return {
            "route_selected": route_decision,
            "result": result
        }

    # ==========================================
    # Helper Methods (Mock implementations)
    # ==========================================
    
    def _route_request(self, image, prompt):
        """
        Passes the image and prompt to the router model.
        It should return a strict structural label.
        """
        # router_prompt = f"Determine the task type for: '{prompt}'. Reply with MATH_EXPERT, OCR_EXPERT, or GENERAL_EXPERT."
        # output = self.router_model.generate(image, router_prompt)
        
        # Mock logic:
        prompt_lower = prompt.lower()
        if "math" in prompt_lower or "equation" in prompt_lower:
            return "MATH_EXPERT"
        elif "read" in prompt_lower or "text" in prompt_lower:
            return "OCR_EXPERT"
        return "GENERAL_EXPERT"

    def _call_math_expert(self, image, prompt):
        # return self.math_expert.generate(image, prompt)
        return "Calculated math result from the Math Expert Sub-model."

    def _call_ocr_expert(self, image, prompt):
        # return self.ocr_expert.generate(image, prompt)
        return "Extracted text string from the OCR Expert Sub-model."

    def _call_general_expert(self, image, prompt):
        # return self.general_expert.generate(image, prompt)
        return "General visual description from the General Sub-model."
