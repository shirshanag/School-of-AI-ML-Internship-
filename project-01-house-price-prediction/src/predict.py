from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd
import os

def main(model_path: str, input_json: str) -> None:
    # 1. Load the model
    pipeline = joblib.load(model_path)
    
    # 2. Logic to handle either a File Path or a Raw JSON String
    if os.path.exists(input_json):
        with open(input_json, 'r') as f:
            row = json.load(f)
    else:
        # If it's not a file, try parsing the string directly
        try:
            row = json.loads(input_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input. If passing a string, check your quotes.")
            print(f"Received: {input_json}")
            return

    # 3. Predict
    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    print(f"Predicted price (target units): {pred:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True, 
                        help="A JSON string or a path to a .json file")
    args = parser.parse_args()

    main(args.model_path, args.input_json)