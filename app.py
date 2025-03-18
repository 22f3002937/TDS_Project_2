import os
import zipfile
import pandas as pd
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

@app.route("/api/", methods=["POST"])
def solve_question():
    """Handles API requests with a question and optional file."""
    question = request.form.get("question")
    file = request.files.get("file")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    extracted_data = None

    if file:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        if file.filename.endswith(".zip"):
            extracted_data = extract_csv_from_zip(file_path)

    # Generate answer using OpenAI's GPT
    prompt = f"Question: {question}\n"
    if extracted_data:
        prompt += f"Extracted Data: {extracted_data}\n"

    answer = get_llm_response(prompt)

    return jsonify({"answer": answer})

def extract_csv_from_zip(zip_path):
    """Extracts CSV data from a ZIP file."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".csv"):
                zip_ref.extract(file_name, "uploads")
                df = pd.read_csv(f"uploads/{file_name}")
                return df.to_string()

    return None

def get_llm_response(prompt):
    """Queries OpenAI GPT for an answer."""
    response = client.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    app.run(debug=True)
