
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import tempfile
import sys
import os

app = FastAPI(title="CodeForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str
    input: str = ""

@app.get("/")
def home():
    return {"message": "Welcome to CodeForge!"}

@app.post("/run")
def run_code(request: CodeRequest):
    filename = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False
        ) as file:
            file.write(request.code)
            filename = file.name

        result = subprocess.run(
            [sys.executable, filename],
            input=request.input,
            capture_output=True,
            text=True,
            timeout=3
        )

        return {
            "output": result.stdout,
            "error": result.stderr,
            "status": "success" if result.returncode == 0 else "error"
        }

    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": "Time limit exceeded!",
            "status": "error"
        }

    except Exception as error:
        return {
            "output": "",
            "error": str(error),
            "status": "error"
        }

    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)
