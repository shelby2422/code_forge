from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import time

app = FastAPI(title="CodeForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://codeforge-pi-eight.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

JUDGE0_URL = "https://ce.judge0.com"


class CodeRequest(BaseModel):
    code: str
    input: str = ""


class SubmitRequest(BaseModel):
    problem_id: int
    code: str


PROBLEMS = [
    {
        "id": 1,
        "title": "Two Number Addition",
        "difficulty": "Easy",
        "description": "Read two integers and print their sum.",
        "example_input": "2 3",
        "example_output": "5",
        "starter": "a, b = map(int, input().split())\n# Write your solution\n",
        "tests": [("2 3", "5"), ("10 20", "30"), ("-2 5", "3")],
    },
    {
        "id": 2,
        "title": "Palindrome Number",
        "difficulty": "Easy",
        "description": "Read a number and print True if it reads the same forwards and backwards; otherwise print False.",
        "example_input": "121",
        "example_output": "True",
        "starter": "n = input().strip()\n# Write your solution\n",
        "tests": [("121", "True"), ("123", "False"), ("1221", "True")],
    },
    {
        "id": 3,
        "title": "Factorial",
        "difficulty": "Easy",
        "description": "Read a non-negative integer and print its factorial.",
        "example_input": "5",
        "example_output": "120",
        "starter": "n = int(input())\n# Write your solution\n",
        "tests": [("5", "120"), ("0", "1"), ("4", "24")],
    },
    {
        "id": 4,
        "title": "Prime Number",
        "difficulty": "Easy",
        "description": "Read an integer and print True if it is prime; otherwise print False.",
        "example_input": "7",
        "example_output": "True",
        "starter": "n = int(input())\n# Write your solution\n",
        "tests": [("7", "True"), ("8", "False"), ("2", "True")],
    },
    {
        "id": 5,
        "title": "Reverse a String",
        "difficulty": "Easy",
        "description": "Read a string and print it in reverse order.",
        "example_input": "hello",
        "example_output": "olleh",
        "starter": "s = input()\n# Write your solution\n",
        "tests": [("hello", "olleh"), ("Python", "nohtyP"), ("abc", "cba")],
    },
    {
        "id": 6,
        "title": "Maximum Element",
        "difficulty": "Easy",
        "description": "Read a count n, followed by n integers. Print the largest integer.",
        "example_input": "5\n1 9 3 7 2",
        "example_output": "9",
        "starter": "n = int(input())\narr = list(map(int, input().split()))\n# Write your solution\n",
        "tests": [
            ("5\n1 9 3 7 2", "9"),
            ("3\n-5 -2 -9", "-2"),
            ("1\n42", "42"),
        ],
    },
    {
        "id": 7,
        "title": "Fibonacci Number",
        "difficulty": "Easy",
        "description": "Read n and print the nth Fibonacci number, using F(0)=0 and F(1)=1.",
        "example_input": "6",
        "example_output": "8",
        "starter": "n = int(input())\n# Write your solution\n",
        "tests": [("6", "8"), ("0", "0"), ("7", "13")],
    },
    {
        "id": 8,
        "title": "Binary Search",
        "difficulty": "Medium",
        "description": "Read n, then n sorted integers, then a target. Print its zero-based index, or -1 if it is absent.",
        "example_input": "5\n1 3 5 7 9\n7",
        "example_output": "3",
        "starter": "n = int(input())\narr = list(map(int, input().split()))\ntarget = int(input())\n# Write your solution\n",
        "tests": [
            ("5\n1 3 5 7 9\n7", "3"),
            ("5\n1 3 5 7 9\n4", "-1"),
            ("1\n10\n10", "0"),
        ],
    },
]


def execute_code(code, stdin):
    try:
        response = requests.post(
            f"{JUDGE0_URL}/submissions/",
            params={"base64_encoded": "false", "wait": "false"},
            json={
                "source_code": code,
                "language_id": 71,
                "stdin": stdin,
            },
            timeout=15,
        )
        response.raise_for_status()
        token = response.json().get("token")

        if not token:
            return {"status": "Error", "output": "", "error": "No submission token returned."}

        for _ in range(15):
            time.sleep(1)
            result_response = requests.get(
                f"{JUDGE0_URL}/submissions/{token}",
                params={"base64_encoded": "false"},
                timeout=15,
            )
            result_response.raise_for_status()
            result = result_response.json()
            status = result.get("status", {})
            if status.get("id") not in [1, 2]:
                return {
                    "status": status.get("description", "Unknown"),
                    "output": result.get("stdout") or "",
                    "error": (
                        result.get("stderr")
                        or result.get("compile_output")
                        or result.get("message")
                        or ""
                    ),
                }

        return {"status": "Processing", "output": "", "error": "Execution timed out."}

    except requests.RequestException:
        return {"status": "Error", "output": "", "error": "Unable to connect to Judge0."}


@app.get("/")
def home():
    return {"message": "Welcome to CodeForge!"}


@app.get("/problems")
def get_problems():
    return [
        {key: value for key, value in problem.items() if key != "tests"}
        for problem in PROBLEMS
    ]


@app.post("/run")
def run_code(request: CodeRequest):
    result = execute_code(request.code, request.input)
    if result["status"] == "Error" and result["error"] == "Unable to connect to Judge0.":
        raise HTTPException(status_code=502, detail=result["error"])
    return {
        "output": result["output"],
        "error": result["error"],
        "status": result["status"],
    }


@app.post("/submit")
def submit_code(request: SubmitRequest):
    problem = next(
        (p for p in PROBLEMS if p["id"] == request.problem_id),
        None,
    )
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")

    results = []
    passed = 0

    for test_input, expected in problem["tests"]:
        result = execute_code(request.code, test_input)
        actual = result["output"].strip()
        success = result["status"] == "Accepted" and actual == expected.strip()

        if success:
            passed += 1

        results.append({
            "passed": success,
            "status": result["status"] if result["status"] != "Accepted" else (
                "Accepted" if success else "Wrong Answer"
            ),
            "expected": expected,
            "actual": actual,
            "error": result["error"],
        })

        if result["status"] in ["Error", "Processing"]:
            break

    return {
        "status": "Accepted" if passed == len(problem["tests"]) else "Wrong Answer",
        "passed": passed,
        "total": len(problem["tests"]),
        "results": results,
    }
