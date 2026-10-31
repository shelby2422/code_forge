
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import time

app = FastAPI(title="CodeForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://codeforge-pi-eight.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JUDGE0_URL = "https://ce.judge0.com"

class CodeRequest(BaseModel):
    code: str
    input: str = ""

@app.get("/")
def home():
    return {"message": "Welcome to CodeForge!"}

@app.post("/run")
def run_code(request: CodeRequest):
    try:
        # Submit code to Judge0
        response = requests.post(
            f"{JUDGE0_URL}/submissions/",
            params={"base64_encoded": "false", "wait": "false"},
            json={
                "source_code": request.code,
                "language_id": 71,
                "stdin": request.input
            },
            timeout=15
        )

        response.raise_for_status()
        token = response.json().get("token")

        if not token:
            raise HTTPException(
                status_code=502,
                detail="Judge0 did not return a submission token"
            )

        # Check execution result
        for _ in range(15):
            time.sleep(1)

            result_response = requests.get(
                f"{JUDGE0_URL}/submissions/{token}",
                params={"base64_encoded": "false"},
                timeout=15
            )

            result_response.raise_for_status()
            result = result_response.json()

            status_id = result.get("status", {}).get("id")

            if status_id not in [1, 2]:
                return {
                    "output": result.get("stdout") or "",
                    "error": (
                        result.get("stderr")
                        or result.get("compile_output")
                        or result.get("message")
                        or ""
                    ),
                    "status": result.get("status", {}).get(
                        "description", "Unknown"
                    )
                }

        return {
            "output": "",
            "error": "Execution is taking too long. Please try again.",
            "status": "Processing"
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Unable to connect to Judge0. Please try again."
        )
