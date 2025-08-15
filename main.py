import tempfile
import zipfile
import tarfile
import os
import subprocess
import sys
import json
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import base64
import io
from dotenv import load_dotenv
import openai

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

def get_content_type_for_image(filename):
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "gif":
        return "image/gif"
    else:
        return "application/octet-stream"

class FileData:
    def __init__(self, name, content, content_type, is_image=False, is_text=False):
        self.name = name
        self.content = content
        self.content_type = content_type
        self.is_image = is_image
        self.is_text = is_text

@app.post("/api/")
async def analyze_data(request: Request):
    form = await request.form()
    
    if "questions.txt" not in form:
        return JSONResponse(status_code=400, content={"message": "questions.txt is required"})

    questions_content = ""
    processed_files = []
    files_info_response = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a directory to hold the uploaded and extracted files for the subprocess
        file_storage_path = os.path.join(temp_dir, "files")
        os.makedirs(file_storage_path, exist_ok=True)

        for name, file_or_field in form.items():
            if not hasattr(file_or_field, 'filename'):
                files_info_response[name] = file_or_field
                continue

            file = file_or_field
            content = await file.read()

            if name == "questions.txt":
                questions_content = content.decode('utf-8')
                files_info_response[name] = {"filename": file.filename, "content_preview": questions_content[:200]}
                continue

            is_zip = file.filename.lower().endswith('.zip')
            is_tar = file.filename.lower().endswith(('.tar', '.tar.gz', '.tgz'))

            if is_zip or is_tar:
                files_info_response[name] = {"filename": file.filename, "extracted_files": []}
                archive_path = io.BytesIO(content)
                try:
                    if is_zip:
                        with zipfile.ZipFile(archive_path) as zf:
                            zf.extractall(file_storage_path)
                    elif is_tar:
                        with tarfile.open(fileobj=archive_path) as tf:
                            tf.extractall(file_storage_path)
                    
                    for extracted_filename in os.listdir(file_storage_path):
                        extracted_filepath = os.path.join(file_storage_path, extracted_filename)
                        with open(extracted_filepath, 'rb') as f:
                            extracted_content = f.read()
                        
                        is_image = extracted_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
                        content_type = get_content_type_for_image(extracted_filename) if is_image else 'text/plain'
                        
                        processed_files.append(FileData(
                            name=extracted_filename,
                            content=extracted_content,
                            content_type=content_type,
                            is_image=is_image,
                            is_text=not is_image
                        ))
                        files_info_response[name]["extracted_files"].append({"filename": extracted_filename})

                except (zipfile.BadZipFile, tarfile.ReadError) as e:
                    files_info_response[name]["error"] = f"Could not decompress file: {e}"
            else:
                is_image = file.content_type and file.content_type.startswith('image/')
                
                # Save the file to the temporary directory
                file_path = os.path.join(file_storage_path, file.filename)
                with open(file_path, 'wb') as f:
                    f.write(content)

                processed_files.append(FileData(
                    name=file.filename,
                    content=content,
                    content_type=file.content_type,
                    is_image=is_image,
                    is_text=not is_image
                ))
                files_info_response[name] = {"filename": file.filename}

        prompt_messages = []
        if questions_content:
            prompt_messages.append({"type": "text", "text": f"Here is the question I need you to answer:\n---\n{questions_content}\n---"})
            prompt_messages.append({"type": "text", "text": "To answer the question, you have access to the following files:"})

            for p_file in processed_files:
                if p_file.is_image:
                    encoded_image = base64.b64encode(p_file.content).decode('utf-8')
                    prompt_messages.append({"type": "text", "text": f"- An image file named `{p_file.name}`."})
                    prompt_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{p_file.content_type};base64,{encoded_image}"}
                    })
                elif p_file.is_text:
                    try:
                        decoded_content = p_file.content.decode('utf-8')
                        preview = "\n".join(decoded_content.splitlines()[:15])
                        prompt_messages.append({"type": "text", "text": f"- A text file named `{p_file.name}`. Here is a preview of its content:\n---\n{preview}\n---"})
                    except UnicodeDecodeError:
                        prompt_messages.append({"type": "text", "text": f"- A file named `{p_file.name}` which appears to be binary."})

        
        generated_code = "# No valid question provided or an error occurred."
        if prompt_messages:
            system_prompt = (
                "You are a world-class data analyst AI. Your purpose is to write robust, production-quality Python code to solve a user's question based on the data files they provide. "
                "You must follow these instructions meticulously:\n"
                "1.  **Analyze the Request:** Carefully read the user's question and examine the previews of all provided files (text, CSV, images, etc.) to understand the context and requirements fully."
                "2.  **Think Step-by-Step:** Before writing code, formulate a clear plan. Consider data loading, necessary cleaning (handling missing values, correcting data types, ensuring case consistency), analysis steps, and the final output format."
                "3.  **Write High-Quality Python Code:\n"
                "    - The code must be pure Python and executable. Assume standard libraries like `pandas`, `matplotlib`, `numpy`, and `base64` are available.\n"
                "    - Refer to files by their exact filenames as provided in the prompt (e.g., `sample-sales.csv`). Do not invent or assume file paths.\n"
                "    - **Crucial:** Perform data cleaning and preprocessing. Do not make assumptions about data quality. Check for and handle inconsistencies.\n"
                "    - Your code must print the final answer(s) to standard output. The output format must precisely match what the user requested.\n"
                "    - If the question requires creating a plot or image, you MUST save it to a file (e.g., `plot.png`) and then print its base64 data URI to standard output (e.g., `print(f'data:image/png;base64,{base64_string}')`).\n"
                "4.  **Final Output:** Your response MUST contain ONLY the raw Python code. Do not include any explanations, comments, or markdown formatting like ```python ... ```. Just the code itself."
            )
            
            client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_messages},
                    ],
                )
                generated_code = response.choices[0].message.content
            except Exception as e:
                generated_code = f"# An error occurred while contacting the OpenAI API: {e}"

        # New: Execute the generated code and capture the output
        if generated_code.startswith("#"):
            # The LLM failed to generate code, or a known error occurred
            return JSONResponse(status_code=500, content={"error": generated_code})
        
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(generated_code)

        try:
            # Run the script in the directory where files are stored
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=file_storage_path, # Set the working directory
                capture_output=True,
                text=True,
                check=True
            )
            raw_output = proc.stdout.strip()
            # Attempt to parse the output as JSON
            final_answer = json.loads(raw_output)
            return JSONResponse(content=final_answer)

        except subprocess.CalledProcessError as e:
            # The script failed to run
            error_details = f"Script execution failed. Stderr: {e.stderr}"
            return JSONResponse(status_code=500, content={"error": error_details})

        except json.JSONDecodeError:
            # The script ran, but the output was not valid JSON
            error_details = f"Script produced invalid JSON. Raw output: '{raw_output}'"
            return JSONResponse(status_code=500, content={"error": error_details})
            
        except Exception as e:
            # Catch any other unexpected errors during execution
            return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred during execution: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
