"""
CodeForge backend (FastAPI + Judge0 Community Edition)
======================================================

Endpoints
    GET  /                 -> service info
    GET  /health           -> tiny health check (does NOT call Judge0)
    GET  /judge0-check     -> checks that Judge0 is reachable (use it when debugging)
    GET  /languages        -> supported languages
    GET  /problems         -> all problems (WITHOUT hidden test data) + starter code per language
    POST /run              -> run code once with custom input
    POST /submit           -> run code against every stored test case of one problem

All code is executed by Judge0 (https://ce.judge0.com) inside its own sandbox.
This server never runs student code itself.

Optional environment variables (none are required):
    JUDGE0_URL            default https://ce.judge0.com
    JUDGE0_AUTH_TOKEN     only if you use a private Judge0 that needs a token
    CORS_ORIGINS          extra allowed origins, comma separated
    RATE_LIMIT_PER_MIN    default 30 (per IP, for /run and /submit)
"""

import asyncio
import base64
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
JUDGE0_URL = os.getenv("JUDGE0_URL", "https://ce.judge0.com").rstrip("/")
JUDGE0_AUTH_TOKEN = os.getenv("JUDGE0_AUTH_TOKEN", "")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

ALLOWED_ORIGINS = [
    "https://codeforge-pi-eight.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
ALLOWED_ORIGINS += [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

MAX_CODE_CHARS = 20_000
MAX_STDIN_CHARS = 10_000
MAX_OUTPUT_CHARS = 20_000
POLL_DEADLINE_SECONDS = 60
MAX_PARALLEL_JUDGE0_JOBS = 6

# Judge0 language ids -> settings. memory_limit is in KB, times are in seconds.
LANGUAGES: Dict[int, Dict[str, Any]] = {
    71: {"name": "Python 3", "monaco": "python", "cpu": 3.0, "wall": 10.0, "memory": 256000},
    50: {"name": "C (GCC)", "monaco": "c", "cpu": 2.0, "wall": 10.0, "memory": 256000},
    54: {"name": "C++ (G++)", "monaco": "cpp", "cpu": 2.0, "wall": 10.0, "memory": 256000},
    62: {"name": "Java", "monaco": "java", "cpu": 4.0, "wall": 15.0, "memory": 512000},
    63: {"name": "JavaScript (Node.js)", "monaco": "javascript", "cpu": 3.0, "wall": 10.0, "memory": 256000},
}

# Judge0 status ids -> (short verdict shown to the student)
VERDICTS = {
    3: "Accepted",
    4: "Wrong Answer",
    5: "Time Limit Exceeded",
    6: "Compilation Error",
    7: "Runtime Error",
    8: "Runtime Error",
    9: "Runtime Error",
    10: "Runtime Error",
    11: "Runtime Error",
    12: "Runtime Error",
    13: "Internal Error",
    14: "Exec Format Error",
}

# ----------------------------------------------------------------------------
# Problem bank
#
# spec   : how the input is laid out (used to build starter code for every language)
#            ["int", name]          one integer on its own line
#            ["pair", [n1, n2]]     several integers on one line
#            ["array", name]        a line with the length n, then a line with n integers
#            ["word", name]         one word (no spaces) on its own line
#            ["line", name]         a whole line of text (may contain spaces)
# out    : kind of answer printed (int / text / arr / lines) - used for starter code
# examples: shown to students AND used as the first test cases of /submit
# hidden : extra test cases, never sent to the browser
# ----------------------------------------------------------------------------
PROBLEMS = [
    {
        "id": 1,
        "title": "Two Number Addition",
        "difficulty": "Easy",
        "tags": ["Math", "Basics"],
        "description": "Given two integers `a` and `b`, print their sum.",
        "input_format": "One line containing two space-separated integers `a` and `b`.",
        "output_format": "Print a single integer: `a + b`.",
        "constraints": ["-10^9 <= a, b <= 10^9"],
        "spec": [["pair", ["a", "b"]]],
        "out": "int",
        "examples": [
            {"input": "3 5", "output": "8",
             "explanation": "3 + 5 = 8."},
            {"input": "-2 10", "output": "8",
             "explanation": "-2 + 10 = 8."},
        ],
        "hidden": [
            ["0 0", "0"],
            ["-7 -8", "-15"],
            ["1000000000 1000000000", "2000000000"],
            ["123456 654321", "777777"],
            ["-1000000000 999999999", "-1"],
        ],
    },
    {
        "id": 2,
        "title": "Palindrome Number",
        "difficulty": "Easy",
        "tags": ["Math", "Number Manipulation"],
        "description": "A palindrome number reads the same from left to right and from right to left. Given a non-negative integer `n`, print `Yes` if it is a palindrome, otherwise print `No`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print `Yes` or `No` (exactly like that, with a capital first letter).",
        "constraints": ["0 <= n <= 10^9"],
        "spec": [["int", "n"]],
        "out": "text",
        "examples": [
            {"input": "121", "output": "Yes",
             "explanation": "121 reads the same in both directions."},
            {"input": "123", "output": "No",
             "explanation": "123 reversed is 321, which is different."},
        ],
        "hidden": [
            ["0", "Yes"],
            ["7", "Yes"],
            ["10", "No"],
            ["1221", "Yes"],
            ["1000021", "No"],
            ["123454321", "Yes"],
            ["120", "No"],
        ],
    },
    {
        "id": 3,
        "title": "Factorial",
        "difficulty": "Easy",
        "tags": ["Math", "Loops"],
        "description": "The factorial of `n` (written `n!`) is the product of all positive integers from 1 to `n`. By definition `0! = 1`. Given `n`, print `n!`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print a single integer: `n!`.",
        "constraints": ["0 <= n <= 12 (the answer fits in a 32-bit integer)"],
        "spec": [["int", "n"]],
        "out": "int",
        "examples": [
            {"input": "5", "output": "120",
             "explanation": "5! = 5 x 4 x 3 x 2 x 1 = 120."},
            {"input": "0", "output": "1",
             "explanation": "By definition 0! = 1."},
        ],
        "hidden": [
            ["1", "1"],
            ["3", "6"],
            ["7", "5040"],
            ["10", "3628800"],
            ["12", "479001600"],
        ],
    },
    {
        "id": 4,
        "title": "Prime Number",
        "difficulty": "Easy",
        "tags": ["Math", "Number Theory"],
        "description": "A prime number is greater than 1 and has no divisors other than 1 and itself. Given `n`, print `Prime` if it is a prime number, otherwise print `Not Prime`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print `Prime` or `Not Prime`.",
        "constraints": ["1 <= n <= 10^9", "Checking divisors only up to the square root of n is fast enough."],
        "spec": [["int", "n"]],
        "out": "text",
        "examples": [
            {"input": "7", "output": "Prime",
             "explanation": "7 is only divisible by 1 and 7."},
            {"input": "12", "output": "Not Prime",
             "explanation": "12 is divisible by 2, 3, 4 and 6."},
        ],
        "hidden": [
            ["1", "Not Prime"],
            ["2", "Prime"],
            ["4", "Not Prime"],
            ["97", "Prime"],
            ["100", "Not Prime"],
            ["7919", "Prime"],
            ["999999937", "Prime"],
            ["1000000000", "Not Prime"],
        ],
    },
    {
        "id": 5,
        "title": "Reverse a String",
        "difficulty": "Easy",
        "tags": ["Strings"],
        "description": "Given a line of text `s`, print it reversed. Spaces, digits and punctuation are reversed too.",
        "input_format": "One line containing the text `s` (it may contain spaces).",
        "output_format": "Print the reversed text.",
        "constraints": ["1 <= length of s <= 1000", "s does not start or end with a space."],
        "spec": [["line", "s"]],
        "out": "text",
        "examples": [
            {"input": "hello", "output": "olleh",
             "explanation": "Reading hello backwards gives olleh."},
            {"input": "CodeForge 2026", "output": "6202 egroFedoC",
             "explanation": "The space and the digits are reversed as well."},
        ],
        "hidden": [
            ["a", "a"],
            ["racecar", "racecar"],
            ["abc def", "fed cba"],
            ["Hello, World!", "!dlroW ,olleH"],
            ["12345", "54321"],
            ["The quick brown fox", "xof nworb kciuq ehT"],
        ],
    },
    {
        "id": 6,
        "title": "Maximum Element",
        "difficulty": "Easy",
        "tags": ["Arrays"],
        "description": "Given an array of `n` integers, print the largest element.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the maximum element.",
        "constraints": ["1 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "int",
        "examples": [
            {"input": "5\n3 1 4 1 5", "output": "5",
             "explanation": "The largest value is 5."},
            {"input": "3\n-5 -2 -9", "output": "-2",
             "explanation": "All values are negative; -2 is the largest."},
        ],
        "hidden": [
            ["1\n7", "7"],
            ["5\n1 2 3 4 5", "5"],
            ["5\n5 4 3 2 1", "5"],
            ["3\n-1 -1 -1", "-1"],
            ["5\n0 -3 8 8 2", "8"],
            ["50\n-750188 -329993 53851 995109 73370 357272 803551 998482 -784755 854133 -531739 876546 260452 303346 167188 -117907 642853 200763 148843 766644 532051 628590 610854 29501 575015 621251 230023 -74957 -496822 -994685 288079 -830711 -767769 -397614 712872 -794371 -57006 -975907 708681 868486 436922 28205 424781 -341006 -559127 -166975 -472539 -270909 948919 -252080", "998482"],
        ],
    },
    {
        "id": 7,
        "title": "Fibonacci Number",
        "difficulty": "Easy",
        "tags": ["Math", "Loops", "Dynamic Programming"],
        "description": "The Fibonacci sequence starts `F(0) = 0`, `F(1) = 1`, and every next number is the sum of the previous two: `F(n) = F(n-1) + F(n-2)`. Given `n`, print `F(n)`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print a single integer: `F(n)`.",
        "constraints": ["0 <= n <= 40"],
        "spec": [["int", "n"]],
        "out": "int",
        "examples": [
            {"input": "10", "output": "55",
             "explanation": "The sequence is 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 so F(10) = 55."},
            {"input": "1", "output": "1",
             "explanation": "F(1) = 1."},
        ],
        "hidden": [
            ["0", "0"],
            ["2", "1"],
            ["20", "6765"],
            ["30", "832040"],
            ["40", "102334155"],
        ],
    },
    {
        "id": 8,
        "title": "Binary Search",
        "difficulty": "Medium",
        "tags": ["Searching", "Arrays", "Binary Search"],
        "description": "You are given an array of `n` distinct integers sorted in ascending order and a `target`. Print the 0-based index of `target` in the array, or `-1` if it is not present. Try to solve it in O(log n) time with binary search.",
        "input_format": "The first line contains `n`. The second line contains `n` sorted integers. The third line contains `target`.",
        "output_format": "Print the index of `target`, or `-1`.",
        "constraints": ["1 <= n <= 1000", "All array values are distinct and sorted in ascending order."],
        "spec": [["array", "arr"], ["int", "target"]],
        "out": "int",
        "examples": [
            {"input": "6\n1 3 5 7 9 11\n7", "output": "3",
             "explanation": "7 is at index 3 (0-based)."},
            {"input": "4\n2 4 6 8\n5", "output": "-1",
             "explanation": "5 is not in the array, so print -1."},
        ],
        "hidden": [
            ["1\n10\n10", "0"],
            ["1\n10\n5", "-1"],
            ["5\n1 2 3 4 5\n1", "0"],
            ["5\n1 2 3 4 5\n5", "4"],
            ["6\n-8 -3 0 4 9 15\n0", "2"],
            ["100\n-475 -450 -422 -409 -408 -395 -381 -376 -368 -356 -355 -271 -266 -243 -236 -206 -201 -185 -184 -152 -128 -123 -115 -93 -92 -89 -70 -68 -65 -61 -57 -33 -32 -28 -20 -19 -10 -7 2 10 13 18 25 30 35 37 50 58 61 67 70 73 77 89 91 99 114 137 141 153 160 162 185 186 190 215 217 220 221 227 231 233 234 240 263 268 284 318 321 324 337 364 368 370 387 388 396 397 405 407 408 411 412 438 444 447 448 478 496 499\n186", "63"],
            ["100\n-475 -450 -422 -409 -408 -395 -381 -376 -368 -356 -355 -271 -266 -243 -236 -206 -201 -185 -184 -152 -128 -123 -115 -93 -92 -89 -70 -68 -65 -61 -57 -33 -32 -28 -20 -19 -10 -7 2 10 13 18 25 30 35 37 50 58 61 67 70 73 77 89 91 99 114 137 141 153 160 162 185 186 190 215 217 220 221 227 231 233 234 240 263 268 284 318 321 324 337 364 368 370 387 388 396 397 405 407 408 411 412 438 444 447 448 478 496 499\n1000", "-1"],
        ],
    },
    {
        "id": 9,
        "title": "Even or Odd",
        "difficulty": "Easy",
        "tags": ["Math", "Basics"],
        "description": "Given an integer `n`, print `Even` if it is divisible by 2, otherwise print `Odd`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print `Even` or `Odd`.",
        "constraints": ["-10^9 <= n <= 10^9", "Negative numbers can be even or odd too."],
        "spec": [["int", "n"]],
        "out": "text",
        "examples": [
            {"input": "4", "output": "Even",
             "explanation": "4 is divisible by 2."},
            {"input": "7", "output": "Odd",
             "explanation": "7 leaves remainder 1 when divided by 2."},
        ],
        "hidden": [
            ["0", "Even"],
            ["-3", "Odd"],
            ["-8", "Even"],
            ["1000000000", "Even"],
            ["999999999", "Odd"],
        ],
    },
    {
        "id": 10,
        "title": "Sum of First N Natural Numbers",
        "difficulty": "Easy",
        "tags": ["Math", "Loops"],
        "description": "Given `n`, print the sum `1 + 2 + 3 + ... + n`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print a single integer: the sum.",
        "constraints": ["1 <= n <= 10000"],
        "spec": [["int", "n"]],
        "out": "int",
        "examples": [
            {"input": "5", "output": "15",
             "explanation": "1 + 2 + 3 + 4 + 5 = 15."},
            {"input": "10", "output": "55",
             "explanation": "1 + 2 + ... + 10 = 55."},
        ],
        "hidden": [
            ["1", "1"],
            ["2", "3"],
            ["100", "5050"],
            ["9999", "49995000"],
            ["10000", "50005000"],
        ],
    },
    {
        "id": 11,
        "title": "Greatest Common Divisor",
        "difficulty": "Easy",
        "tags": ["Math", "Number Theory"],
        "description": "Given two positive integers `a` and `b`, print their greatest common divisor (GCD): the largest integer that divides both numbers.",
        "input_format": "One line containing two space-separated integers `a` and `b`.",
        "output_format": "Print a single integer: `gcd(a, b)`.",
        "constraints": ["1 <= a, b <= 10^9", "The Euclidean algorithm is fast enough."],
        "spec": [["pair", ["a", "b"]]],
        "out": "int",
        "examples": [
            {"input": "12 18", "output": "6",
             "explanation": "The divisors of 12 and 18 in common are 1, 2, 3, 6. The largest is 6."},
            {"input": "7 13", "output": "1",
             "explanation": "7 and 13 only share the divisor 1."},
        ],
        "hidden": [
            ["100 25", "25"],
            ["17 17", "17"],
            ["1000000000 500000000", "500000000"],
            ["48 180", "12"],
            ["1 999999999", "1"],
            ["270 192", "6"],
        ],
    },
    {
        "id": 12,
        "title": "Leap Year",
        "difficulty": "Easy",
        "tags": ["Math", "Conditions"],
        "description": "A year is a leap year if it is divisible by 4, except years divisible by 100 are not leap years, unless they are also divisible by 400. Given a `year`, print `Leap Year` or `Not a Leap Year`.",
        "input_format": "A single integer `year`.",
        "output_format": "Print `Leap Year` or `Not a Leap Year`.",
        "constraints": ["1 <= year <= 9999"],
        "spec": [["int", "year"]],
        "out": "text",
        "examples": [
            {"input": "2024", "output": "Leap Year",
             "explanation": "2024 is divisible by 4 and not by 100."},
            {"input": "2023", "output": "Not a Leap Year",
             "explanation": "2023 is not divisible by 4."},
        ],
        "hidden": [
            ["1900", "Not a Leap Year"],
            ["2000", "Leap Year"],
            ["2100", "Not a Leap Year"],
            ["1600", "Leap Year"],
            ["4", "Leap Year"],
            ["1", "Not a Leap Year"],
            ["2400", "Leap Year"],
        ],
    },
    {
        "id": 13,
        "title": "Sum of Digits",
        "difficulty": "Easy",
        "tags": ["Math", "Number Manipulation"],
        "description": "Given a non-negative integer `n`, print the sum of its digits.",
        "input_format": "A single integer `n`.",
        "output_format": "Print a single integer: the sum of the digits of `n`.",
        "constraints": ["0 <= n <= 10^9"],
        "spec": [["int", "n"]],
        "out": "int",
        "examples": [
            {"input": "123", "output": "6",
             "explanation": "1 + 2 + 3 = 6."},
            {"input": "9045", "output": "18",
             "explanation": "9 + 0 + 4 + 5 = 18."},
        ],
        "hidden": [
            ["0", "0"],
            ["7", "7"],
            ["1000000000", "1"],
            ["999999999", "81"],
            ["56789", "35"],
        ],
    },
    {
        "id": 14,
        "title": "Reverse a Number",
        "difficulty": "Easy",
        "tags": ["Math", "Number Manipulation"],
        "description": "Given a non-negative integer `n`, print the number obtained by reversing its digits. Leading zeros of the result are dropped (for example `120` becomes `21`).",
        "input_format": "A single integer `n`.",
        "output_format": "Print a single integer: the reversed number.",
        "constraints": ["0 <= n <= 10^9"],
        "spec": [["int", "n"]],
        "out": "int",
        "examples": [
            {"input": "1234", "output": "4321",
             "explanation": "The digits 1234 reversed are 4321."},
            {"input": "120", "output": "21",
             "explanation": "Reversed digits are 021, which is the number 21."},
        ],
        "hidden": [
            ["0", "0"],
            ["5", "5"],
            ["1000", "1"],
            ["987654321", "123456789"],
            ["1000000000", "1"],
            ["10203", "30201"],
        ],
    },
    {
        "id": 15,
        "title": "Armstrong Number",
        "difficulty": "Easy",
        "tags": ["Math", "Number Manipulation"],
        "description": "A number is an Armstrong number if it equals the sum of its own digits, each raised to the power of the number of digits. For example `153 = 1^3 + 5^3 + 3^3`. Given `n`, print `Yes` if it is an Armstrong number, otherwise `No`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print `Yes` or `No`.",
        "constraints": ["1 <= n <= 10^6"],
        "spec": [["int", "n"]],
        "out": "text",
        "examples": [
            {"input": "153", "output": "Yes",
             "explanation": "1^3 + 5^3 + 3^3 = 1 + 125 + 27 = 153."},
            {"input": "123", "output": "No",
             "explanation": "1^3 + 2^3 + 3^3 = 36, which is not 123."},
        ],
        "hidden": [
            ["9", "Yes"],
            ["10", "No"],
            ["370", "Yes"],
            ["9474", "Yes"],
            ["9475", "No"],
            ["548834", "Yes"],
            ["100000", "No"],
        ],
    },
    {
        "id": 16,
        "title": "Sum of Array",
        "difficulty": "Easy",
        "tags": ["Arrays", "Loops"],
        "description": "Given an array of `n` integers, print the sum of all its elements.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the sum of the array.",
        "constraints": ["1 <= n <= 1000", "-1000 <= array[i] <= 1000"],
        "spec": [["array", "arr"]],
        "out": "int",
        "examples": [
            {"input": "5\n1 2 3 4 5", "output": "15",
             "explanation": "1 + 2 + 3 + 4 + 5 = 15."},
            {"input": "3\n10 -4 7", "output": "13",
             "explanation": "10 - 4 + 7 = 13."},
        ],
        "hidden": [
            ["1\n0", "0"],
            ["1\n-5", "-5"],
            ["3\n1000 1000 1000", "3000"],
            ["4\n1 -1 1 -1", "0"],
            ["100\n-586 -398 703 -450 -982 -933 -694 342 806 223 967 -62 645 70 898 2 -263 887 -591 -468 418 -6 -45 47 570 692 18 654 -111 142 359 858 -932 702 -406 -352 -63 877 662 -799 -103 411 467 607 970 55 -470 -73 888 976 -626 980 -129 725 490 -19 925 -869 328 -455 849 20 -242 -803 -602 530 754 -158 -701 -80 876 -972 -812 -1000 -76 189 119 909 712 -830 249 625 -593 -493 -753 271 -909 539 119 958 -451 369 551 301 -269 -684 -990 -359 361 194", "6197"],
        ],
    },
    {
        "id": 17,
        "title": "Second Largest Element",
        "difficulty": "Easy",
        "tags": ["Arrays"],
        "description": "Given an array of `n` integers, print the second largest distinct value. The array is guaranteed to contain at least two different values.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the second largest distinct value.",
        "constraints": ["2 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "int",
        "examples": [
            {"input": "5\n3 1 4 1 5", "output": "4",
             "explanation": "The largest is 5 and the second largest is 4."},
            {"input": "3\n10 10 9", "output": "9",
             "explanation": "The distinct values are 10 and 9, so the answer is 9."},
        ],
        "hidden": [
            ["2\n1 2", "1"],
            ["4\n5 5 5 4", "4"],
            ["3\n-1 -2 -3", "-2"],
            ["4\n9 8 7 6", "8"],
            ["4\n1 100 100 50", "50"],
        ],
    },
    {
        "id": 18,
        "title": "Reverse an Array",
        "difficulty": "Easy",
        "tags": ["Arrays"],
        "description": "Given an array of `n` integers, print the elements in reverse order.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print the reversed array as space-separated integers on one line.",
        "constraints": ["1 <= n <= 1000"],
        "spec": [["array", "arr"]],
        "out": "arr",
        "examples": [
            {"input": "5\n1 2 3 4 5", "output": "5 4 3 2 1",
             "explanation": "The array read from the end is 5 4 3 2 1."},
            {"input": "3\n10 20 30", "output": "30 20 10",
             "explanation": "30 20 10."},
        ],
        "hidden": [
            ["1\n1", "1"],
            ["2\n1 2", "2 1"],
            ["3\n-1 0 1", "1 0 -1"],
            ["50\n47 96 25 76 84 -66 39 -43 21 50 -67 53 77 29 -26 -79 6 71 -60 -66 100 100 50 -22 29 5 -1 -62 33 -50 -4 -97 -73 -1 -82 -87 -12 65 -10 97 -94 -94 30 21 -17 74 -23 15 -17 -69", "-69 -17 15 -23 74 -17 21 30 -94 -94 97 -10 65 -12 -87 -82 -1 -73 -97 -4 -50 33 -62 -1 5 29 -22 50 100 100 -66 -60 71 6 -79 -26 29 77 53 -67 50 21 -43 39 -66 84 76 25 96 47"],
            ["4\n7 7 7 7", "7 7 7 7"],
        ],
    },
    {
        "id": 19,
        "title": "Count Even Numbers",
        "difficulty": "Easy",
        "tags": ["Arrays", "Math"],
        "description": "Given an array of `n` integers, print how many of them are even.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the count of even numbers.",
        "constraints": ["1 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "int",
        "examples": [
            {"input": "6\n1 2 3 4 5 6", "output": "3",
             "explanation": "2, 4 and 6 are even."},
            {"input": "3\n7 9 11", "output": "0",
             "explanation": "There are no even numbers."},
        ],
        "hidden": [
            ["1\n0", "1"],
            ["2\n-2 -4", "2"],
            ["4\n2 2 2 2", "4"],
            ["4\n-3 -1 1 3", "0"],
            ["100\n19 943 -45 170 266 648 -651 -685 210 64 -577 -112 276 -312 -61 225 -523 389 455 -858 120 -183 -931 -600 325 887 -645 -908 327 115 906 718 -902 -75 777 545 312 -614 -10 456 389 797 -959 -736 322 85 -553 608 807 58 832 959 -718 -672 354 729 -620 -202 -442 -129 91 -152 777 -839 -216 -376 -617 -296 -643 -461 -50 19 597 669 -149 -411 -43 -638 -333 63 -13 -90 -755 79 631 74 631 -440 671 -807 736 -562 -527 -990 953 -810 799 -70 322 -324", "46"],
        ],
    },
    {
        "id": 20,
        "title": "Move Zeroes",
        "difficulty": "Easy",
        "tags": ["Arrays", "Two Pointers"],
        "description": "Given an array of `n` integers, move all `0`s to the end while keeping the relative order of the non-zero elements. Print the resulting array.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print the resulting array as space-separated integers on one line.",
        "constraints": ["1 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "arr",
        "examples": [
            {"input": "5\n0 1 0 3 12", "output": "1 3 12 0 0",
             "explanation": "The non-zero numbers 1 3 12 stay in order, the two zeros go to the end."},
            {"input": "3\n1 2 3", "output": "1 2 3",
             "explanation": "There are no zeros, so nothing changes."},
        ],
        "hidden": [
            ["1\n0", "0"],
            ["3\n0 0 0", "0 0 0"],
            ["6\n4 0 5 0 6 0", "4 5 6 0 0 0"],
            ["3\n-1 0 -2", "-1 -2 0"],
            ["3\n0 0 1", "1 0 0"],
            ["60\n2 3 -4 0 0 1 0 1 3 -4 3 3 0 0 3 -4 1 3 3 -4 0 3 1 3 1 -4 3 0 0 1 0 2 0 0 -4 0 1 -4 0 -4 -4 0 3 1 1 3 0 1 1 0 1 1 0 0 2 1 2 0 0 0", "2 3 -4 1 1 3 -4 3 3 3 -4 1 3 3 -4 3 1 3 1 -4 3 1 2 -4 1 -4 -4 -4 3 1 1 3 1 1 1 1 2 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"],
        ],
    },
    {
        "id": 21,
        "title": "Count Distinct Elements",
        "difficulty": "Easy",
        "tags": ["Arrays", "Hashing"],
        "description": "Given an array of `n` integers, print how many different values it contains.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the number of distinct values.",
        "constraints": ["1 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "int",
        "examples": [
            {"input": "6\n1 2 2 3 3 3", "output": "3",
             "explanation": "The distinct values are 1, 2 and 3."},
            {"input": "3\n5 5 5", "output": "1",
             "explanation": "Only the value 5 appears."},
        ],
        "hidden": [
            ["1\n1", "1"],
            ["4\n1 2 3 4", "4"],
            ["4\n-1 1 -1 1", "2"],
            ["100\n7 0 9 3 3 21 17 18 29 27 11 10 24 14 19 20 20 22 19 1 29 12 10 19 6 19 14 13 21 28 3 21 11 27 16 21 29 8 12 9 30 26 11 1 23 22 13 28 0 9 22 10 9 23 30 24 25 29 3 9 24 13 10 18 28 0 17 30 10 6 17 1 23 3 5 19 3 3 1 4 16 11 19 20 20 9 0 13 18 4 11 13 12 12 7 24 13 21 7 24", "29"],
            ["7\n0 0 0 0 0 0 1", "2"],
        ],
    },
    {
        "id": 22,
        "title": "Two Sum",
        "difficulty": "Medium",
        "tags": ["Arrays", "Hashing", "Interview"],
        "description": "Given an array of `n` integers and a `target`, find the two different positions `i` and `j` (with `i < j`) such that `array[i] + array[j] = target`. Exactly one such pair exists. Print `i` and `j` (0-based) separated by a space.",
        "input_format": "The first line contains `n`. The second line contains `n` integers. The third line contains `target`.",
        "output_format": "Print the two indices `i j` with `i < j`.",
        "constraints": ["2 <= n <= 1000", "-10^6 <= array[i] <= 10^6", "Exactly one valid pair exists."],
        "spec": [["array", "arr"], ["int", "target"]],
        "out": "text",
        "examples": [
            {"input": "4\n2 7 11 15\n9", "output": "0 1",
             "explanation": "arr[0] + arr[1] = 2 + 7 = 9."},
            {"input": "3\n3 2 4\n6", "output": "1 2",
             "explanation": "arr[1] + arr[2] = 2 + 4 = 6."},
        ],
        "hidden": [
            ["2\n3 3\n6", "0 1"],
            ["4\n1 5 9 14\n23", "2 3"],
            ["5\n-1 -2 -3 -4 -5\n-8", "2 4"],
            ["4\n0 4 3 0\n0", "0 3"],
            ["40\n-372 -313 55 -471 177 234 457 54 200 68 -94 410 -239 305 -349 -267 -477 -484 -217 -10 -73 -230 481 -178 -359 466 141 107 -409 359 -317 66 -462 165 386 484 343 26 339 -229\n-65", "21 33"],
            ["100\n500 -203 -317 121 -465 -365 -915 510 213 356 612 -612 -668 840 -401 -396 108 -416 -269 -726 992 -205 -614 999 529 289 -981 735 240 326 -17 923 -135 -272 663 581 847 590 796 100 -651 478 57 -940 -633 -356 104 -704 -665 -942 816 -762 54 367 806 442 317 570 -996 724 -233 -870 -247 -381 -558 89 -685 227 249 -492 -741 -783 -332 -8 -781 789 714 488 -851 -220 -463 263 -383 -760 -288 -29 862 21 -753 137 62 -495 697 -266 399 669 -67 879 198 -294\n825", "8 10"],
        ],
    },
    {
        "id": 23,
        "title": "Maximum Subarray Sum",
        "difficulty": "Medium",
        "tags": ["Arrays", "Dynamic Programming", "Interview"],
        "description": "Given an array of `n` integers, find the contiguous (non-empty) subarray with the largest sum and print that sum. This is known as Kadane's algorithm.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the largest possible subarray sum.",
        "constraints": ["1 <= n <= 1000", "-1000 <= array[i] <= 1000"],
        "spec": [["array", "arr"]],
        "out": "int",
        "examples": [
            {"input": "9\n-2 1 -3 4 -1 2 1 -5 4", "output": "6",
             "explanation": "The subarray 4 -1 2 1 has the largest sum, 6."},
            {"input": "3\n1 2 3", "output": "6",
             "explanation": "The whole array: 1 + 2 + 3 = 6."},
        ],
        "hidden": [
            ["1\n-1", "-1"],
            ["3\n-3 -2 -5", "-2"],
            ["3\n5 -10 5", "5"],
            ["5\n8 -19 5 -4 20", "21"],
            ["60\n49 -39 13 49 -11 -17 46 34 -22 -29 -9 5 14 21 -8 -9 42 34 13 1 -10 -28 -25 9 -22 38 -6 47 50 -6 2 25 -2 -23 -31 6 45 22 34 44 -34 5 -16 -8 -35 -10 -16 6 -17 -32 19 30 -16 -21 -8 -32 21 48 -35 -16", "346"],
            ["3\n0 0 0", "0"],
        ],
    },
    {
        "id": 24,
        "title": "Count Vowels",
        "difficulty": "Easy",
        "tags": ["Strings"],
        "description": "Given a line of text, count the vowels in it. The vowels are `a`, `e`, `i`, `o`, `u` in both lower case and upper case.",
        "input_format": "One line containing the text `s` (it may contain spaces and punctuation).",
        "output_format": "Print a single integer: the number of vowels.",
        "constraints": ["1 <= length of s <= 1000", "s does not start or end with a space."],
        "spec": [["line", "s"]],
        "out": "int",
        "examples": [
            {"input": "Hello World", "output": "3",
             "explanation": "The vowels are e, o, o."},
            {"input": "rhythm", "output": "0",
             "explanation": "There are no vowels."},
        ],
        "hidden": [
            ["AEIOU", "5"],
            ["a", "1"],
            ["Programming is fun", "5"],
            ["12345 !!", "0"],
            ["Why? Sky, fly, try.", "0"],
        ],
    },
    {
        "id": 25,
        "title": "Valid Anagram",
        "difficulty": "Easy",
        "tags": ["Strings", "Hashing"],
        "description": "Two words are anagrams if one can be rearranged to form the other using every letter exactly once. Given words `a` and `b`, print `Yes` if they are anagrams, otherwise `No`.",
        "input_format": "The first line contains the word `a`. The second line contains the word `b`.",
        "output_format": "Print `Yes` or `No`.",
        "constraints": ["1 <= length of a, b <= 1000", "Only lowercase English letters."],
        "spec": [["word", "a"], ["word", "b"]],
        "out": "text",
        "examples": [
            {"input": "listen\nsilent", "output": "Yes",
             "explanation": "Both words contain the same letters: e, i, l, n, s, t."},
            {"input": "hello\nworld", "output": "No",
             "explanation": "The letters are different."},
        ],
        "hidden": [
            ["a\na", "Yes"],
            ["ab\na", "No"],
            ["anagram\nnagaram", "Yes"],
            ["rat\ncar", "No"],
            ["aabb\nabab", "Yes"],
            ["abc\nabd", "No"],
        ],
    },
    {
        "id": 26,
        "title": "Count Words",
        "difficulty": "Easy",
        "tags": ["Strings"],
        "description": "Given a sentence, print the number of words in it. Words are separated by one or more spaces.",
        "input_format": "One line containing the sentence `s`.",
        "output_format": "Print a single integer: the number of words.",
        "constraints": ["1 <= length of s <= 1000", "s does not start or end with a space."],
        "spec": [["line", "s"]],
        "out": "int",
        "examples": [
            {"input": "I love coding in Python", "output": "5",
             "explanation": "The words are I, love, coding, in, Python."},
            {"input": "Hello World", "output": "2",
             "explanation": "Two words."},
        ],
        "hidden": [
            ["word", "1"],
            ["a  b   c", "3"],
            ["The quick brown fox jumps over the lazy dog", "9"],
            ["one two", "2"],
            ["CodeForge is a mini online judge", "6"],
        ],
    },
    {
        "id": 27,
        "title": "First Non-Repeating Character",
        "difficulty": "Easy",
        "tags": ["Strings", "Hashing"],
        "description": "Given a word `s`, print the first character that appears exactly once. If every character repeats, print `-1`.",
        "input_format": "A single word `s`.",
        "output_format": "Print the character, or `-1`.",
        "constraints": ["1 <= length of s <= 1000", "Only lowercase English letters."],
        "spec": [["word", "s"]],
        "out": "text",
        "examples": [
            {"input": "leetcode", "output": "l",
             "explanation": "l appears only once and comes first."},
            {"input": "aabb", "output": "-1",
             "explanation": "Every character repeats."},
        ],
        "hidden": [
            ["loveleetcode", "v"],
            ["a", "a"],
            ["abcabc", "-1"],
            ["aabbc", "c"],
            ["zzzzzy", "y"],
        ],
    },
    {
        "id": 28,
        "title": "Valid Parentheses",
        "difficulty": "Medium",
        "tags": ["Strings", "Stack", "Interview"],
        "description": "Given a string made only of the characters `(`, `)`, `[`, `]`, `{` and `}`, print `Yes` if the brackets are balanced and correctly nested, otherwise `No`. Every opening bracket must be closed by the same type of bracket in the correct order.",
        "input_format": "A single line `s` with no spaces.",
        "output_format": "Print `Yes` or `No`.",
        "constraints": ["1 <= length of s <= 1000"],
        "spec": [["word", "s"]],
        "out": "text",
        "examples": [
            {"input": "()[]{}", "output": "Yes",
             "explanation": "Each bracket is closed by its own type in order."},
            {"input": "(]", "output": "No",
             "explanation": "( is closed by ], which is wrong."},
        ],
        "hidden": [
            ["{[()]}", "Yes"],
            ["(((", "No"],
            [")(", "No"],
            ["([)]", "No"],
            ["(()", "No"],
            ["[]{}()(){}[]", "Yes"],
            ["]", "No"],
        ],
    },
    {
        "id": 29,
        "title": "Sort an Array",
        "difficulty": "Easy",
        "tags": ["Sorting", "Arrays"],
        "description": "Given an array of `n` integers, sort it in ascending order and print it. You may use a built-in sort or write your own (bubble sort, selection sort, ...).",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print the sorted array as space-separated integers on one line.",
        "constraints": ["1 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "arr",
        "examples": [
            {"input": "6\n5 2 9 1 5 6", "output": "1 2 5 5 6 9",
             "explanation": "In ascending order: 1 2 5 5 6 9."},
            {"input": "3\n3 2 1", "output": "1 2 3",
             "explanation": "1 2 3."},
        ],
        "hidden": [
            ["1\n1", "1"],
            ["2\n2 1", "1 2"],
            ["4\n-3 5 -1 0", "-3 -1 0 5"],
            ["4\n1 2 3 4", "1 2 3 4"],
            ["5\n9 8 7 6 5", "5 6 7 8 9"],
            ["100\n-399 176 -62 296 -179 -182 -956 125 467 -326 380 -162 -311 -806 390 571 890 13 860 388 735 296 951 722 367 565 119 58 -571 -131 -279 327 -324 -878 890 -182 483 -646 -979 -454 144 437 -484 -733 -316 185 319 252 -151 705 593 642 -26 743 821 -11 -307 -976 -924 550 -550 303 -753 353 744 929 -450 -332 -964 596 -572 619 -64 54 -433 275 290 -244 -839 138 418 -616 103 574 -977 -805 -681 -325 991 -551 931 -51 -578 241 378 -651 444 -367 -828 162", "-979 -977 -976 -964 -956 -924 -878 -839 -828 -806 -805 -753 -733 -681 -651 -646 -616 -578 -572 -571 -551 -550 -484 -454 -450 -433 -399 -367 -332 -326 -325 -324 -316 -311 -307 -279 -244 -182 -182 -179 -162 -151 -131 -64 -62 -51 -26 -11 13 54 58 103 119 125 138 144 162 176 185 241 252 275 290 296 296 303 319 327 353 367 378 380 388 390 418 437 444 467 483 550 565 571 574 593 596 619 642 705 722 735 743 744 821 860 890 890 929 931 951 991"],
        ],
    },
    {
        "id": 30,
        "title": "Check if Array is Sorted",
        "difficulty": "Easy",
        "tags": ["Sorting", "Arrays"],
        "description": "Given an array of `n` integers, print `Yes` if it is sorted in non-decreasing order (each element is greater than or equal to the previous one), otherwise print `No`.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print `Yes` or `No`.",
        "constraints": ["1 <= n <= 1000", "-10^6 <= array[i] <= 10^6"],
        "spec": [["array", "arr"]],
        "out": "text",
        "examples": [
            {"input": "4\n1 2 2 5", "output": "Yes",
             "explanation": "Every element is >= the one before it."},
            {"input": "3\n3 1 2", "output": "No",
             "explanation": "1 comes after 3, so it is not sorted."},
        ],
        "hidden": [
            ["1\n7", "Yes"],
            ["2\n5 4", "No"],
            ["3\n1 1 1", "Yes"],
            ["4\n-5 -3 0 2", "Yes"],
            ["4\n1 3 2 4", "No"],
            ["80\n-478 -472 -463 -455 -445 -434 -408 -390 -364 -364 -311 -297 -277 -264 -244 -241 -239 -223 -220 -195 -189 -175 -166 -151 -149 -145 -126 -124 -91 -80 -73 -71 -23 -18 -9 -8 -6 0 1 21 21 24 25 34 34 45 63 72 95 103 110 121 131 159 160 173 177 178 197 198 228 239 243 249 251 273 284 328 334 356 359 362 373 378 383 404 432 461 466 497", "Yes"],
        ],
    },
    {
        "id": 31,
        "title": "Missing Number",
        "difficulty": "Easy",
        "tags": ["Arrays", "Math", "Interview"],
        "description": "You are given an array of `n` distinct integers taken from the range `0` to `n` (inclusive). Exactly one number from that range is missing. Print the missing number.",
        "input_format": "The first line contains `n`. The second line contains `n` space-separated integers.",
        "output_format": "Print a single integer: the missing number.",
        "constraints": ["1 <= n <= 1000", "All numbers are distinct and between 0 and n."],
        "spec": [["array", "nums"]],
        "out": "int",
        "examples": [
            {"input": "3\n3 0 1", "output": "2",
             "explanation": "The range is 0 to 3 and 2 is missing."},
            {"input": "2\n0 1", "output": "2",
             "explanation": "The range is 0 to 2 and 2 is missing."},
        ],
        "hidden": [
            ["1\n1", "0"],
            ["1\n0", "1"],
            ["9\n9 6 4 2 3 5 7 0 1", "8"],
            ["3\n1 2 3", "0"],
            ["4\n0 1 2 3", "4"],
            ["100\n42 63 41 85 44 47 23 9 72 22 32 68 67 50 81 13 78 27 3 60 30 26 55 66 89 16 77 83 34 99 17 100 95 51 24 19 65 76 93 8 52 53 86 43 36 82 58 57 2 70 64 11 61 88 92 45 91 48 29 10 49 84 40 62 79 31 1 87 14 75 38 28 71 54 46 21 94 97 4 98 74 80 33 15 0 18 73 5 56 90 12 25 35 20 96 59 39 7 69 6", "37"],
        ],
    },
    {
        "id": 32,
        "title": "Integer Square Root",
        "difficulty": "Medium",
        "tags": ["Math", "Binary Search"],
        "description": "Given a non-negative integer `n`, print the integer part of its square root, that is the largest integer `x` such that `x * x <= n`. Do not use floating point square root functions: try binary search instead. Careful: `x * x` can overflow a 32-bit integer.",
        "input_format": "A single integer `n`.",
        "output_format": "Print a single integer: floor(sqrt(n)).",
        "constraints": ["0 <= n <= 10^9"],
        "spec": [["int", "n"]],
        "out": "int",
        "examples": [
            {"input": "16", "output": "4",
             "explanation": "4 x 4 = 16."},
            {"input": "8", "output": "2",
             "explanation": "2 x 2 = 4 and 3 x 3 = 9 is too big, so the answer is 2."},
        ],
        "hidden": [
            ["0", "0"],
            ["1", "1"],
            ["2", "1"],
            ["99", "9"],
            ["1000000000", "31622"],
            ["999999999", "31622"],
            ["46340", "215"],
        ],
    },
    {
        "id": 33,
        "title": "FizzBuzz",
        "difficulty": "Easy",
        "tags": ["Loops", "Conditions", "Interview"],
        "description": "Given `n`, print the numbers from 1 to `n`, one per line. But for multiples of 3 print `Fizz` instead of the number, for multiples of 5 print `Buzz`, and for multiples of both 3 and 5 print `FizzBuzz`.",
        "input_format": "A single integer `n`.",
        "output_format": "Print `n` lines.",
        "constraints": ["1 <= n <= 100"],
        "spec": [["int", "n"]],
        "out": "lines",
        "examples": [
            {"input": "5", "output": "1\n2\nFizz\n4\nBuzz",
             "explanation": "1, 2, Fizz (3), 4, Buzz (5)."},
            {"input": "15", "output": "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz",
             "explanation": "The last line is FizzBuzz because 15 is a multiple of both 3 and 5."},
        ],
        "hidden": [
            ["1", "1"],
            ["3", "1\n2\nFizz"],
            ["10", "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz"],
            ["30", "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n16\n17\nFizz\n19\nBuzz\nFizz\n22\n23\nFizz\nBuzz\n26\nFizz\n28\n29\nFizzBuzz"],
            ["100", "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n16\n17\nFizz\n19\nBuzz\nFizz\n22\n23\nFizz\nBuzz\n26\nFizz\n28\n29\nFizzBuzz\n31\n32\nFizz\n34\nBuzz\nFizz\n37\n38\nFizz\nBuzz\n41\nFizz\n43\n44\nFizzBuzz\n46\n47\nFizz\n49\nBuzz\nFizz\n52\n53\nFizz\nBuzz\n56\nFizz\n58\n59\nFizzBuzz\n61\n62\nFizz\n64\nBuzz\nFizz\n67\n68\nFizz\nBuzz\n71\nFizz\n73\n74\nFizzBuzz\n76\n77\nFizz\n79\nBuzz\nFizz\n82\n83\nFizz\nBuzz\n86\nFizz\n88\n89\nFizzBuzz\n91\n92\nFizz\n94\nBuzz\nFizz\n97\n98\nFizz\nBuzz"],
        ],
    },
]


# ----------------------------------------------------------------------------
# Starter code generation (one generator per language, same input format)
# ----------------------------------------------------------------------------
def _array_name(spec):
    for item in spec:
        if item[0] == "array":
            return item[1]
    return None


def starter_python(spec, out):
    L = []
    for item in spec:
        kind, name = item[0], item[1]
        if kind == "int":
            L.append(f"{name} = int(input())")
        elif kind == "pair":
            L.append(f"{', '.join(name)} = map(int, input().split())")
        elif kind == "array":
            L.append("n = int(input())")
            L.append(f"{name} = list(map(int, input().split()))")
        elif kind == "word":
            L.append(f"{name} = input().strip()")
        elif kind == "line":
            L.append(f"{name} = input()")
    L.append("")
    L.append("# TODO: write your solution here")
    arr = _array_name(spec)
    if out == "int":
        L += ["result = 0", "", "print(result)"]
    elif out == "text":
        L += ['result = ""', "", "print(result)"]
    elif out == "arr":
        L += [f"result = list({arr})", "", "print(*result)"]
    else:
        L += ["for i in range(1, n + 1):", "    print(i)"]
    return "\n".join(L) + "\n"


def starter_c(spec, out):
    L = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "",
         "int main() {"]
    for item in spec:
        kind, name = item[0], item[1]
        if kind == "int":
            L += [f"    int {name};", f'    scanf("%d", &{name});']
        elif kind == "pair":
            L += [f"    int {', '.join(name)};",
                  '    scanf("' + " ".join(["%d"] * len(name)) + '", ' + ", ".join("&" + x for x in name) + ");"]
        elif kind == "array":
            L += ["    int n;", '    scanf("%d", &n);', f"    int {name}[n];",
                  "    for (int i = 0; i < n; i++) {", f'        scanf("%d", &{name}[i]);', "    }"]
        elif kind == "word":
            L += [f"    char {name}[10005];", f'    scanf("%10000s", {name});']
        elif kind == "line":
            L += [f"    char {name}[10005];", f"    fgets({name}, sizeof({name}), stdin);",
                  f'    {name}[strcspn({name}, "\\r\\n")] = 0;']
    L += ["", "    // TODO: write your solution here"]
    arr = _array_name(spec)
    if out == "int":
        L += ["    int result = 0;", "", '    printf("%d\\n", result);']
    elif out == "text":
        L += ['    char result[100] = "";', "", '    printf("%s\\n", result);']
    elif out == "arr":
        L += ["    for (int i = 0; i < n; i++) {", f'        printf("%d ", {arr}[i]);', "    }",
              '    printf("\\n");']
    else:
        L += ["    for (int i = 1; i <= n; i++) {", '        printf("%d\\n", i);', "    }"]
    L += ["    return 0;", "}"]
    return "\n".join(L) + "\n"


def starter_cpp(spec, out):
    L = ["#include <bits/stdc++.h>", "using namespace std;", "", "int main() {",
         "    ios_base::sync_with_stdio(false);", "    cin.tie(nullptr);", ""]
    for item in spec:
        kind, name = item[0], item[1]
        if kind == "int":
            L += [f"    int {name};", f"    cin >> {name};"]
        elif kind == "pair":
            L += [f"    int {', '.join(name)};", "    cin >> " + " >> ".join(name) + ";"]
        elif kind == "array":
            L += ["    int n;", "    cin >> n;", f"    vector<int> {name}(n);",
                  f"    for (int i = 0; i < n; i++) cin >> {name}[i];"]
        elif kind == "word":
            L += [f"    string {name};", f"    cin >> {name};"]
        elif kind == "line":
            L += [f"    string {name};", f"    getline(cin, {name});",
                  f"    if (!{name}.empty() && {name}.back() == '\\r') {name}.pop_back();"]
    L += ["", "    // TODO: write your solution here"]
    arr = _array_name(spec)
    if out == "int":
        L += ["    int result = 0;", "", '    cout << result << "\\n";']
    elif out == "text":
        L += ['    string result = "";', "", '    cout << result << "\\n";']
    elif out == "arr":
        L += ["    for (int i = 0; i < n; i++) {", f'        cout << {arr}[i] << " ";', "    }",
              '    cout << "\\n";']
    else:
        L += ["    for (int i = 1; i <= n; i++) {", '        cout << i << "\\n";', "    }"]
    L += ["    return 0;", "}"]
    return "\n".join(L) + "\n"


def starter_java(spec, out):
    L = ["import java.util.*;", "", "public class Main {",
         "    public static void main(String[] args) {",
         "        Scanner sc = new Scanner(System.in);"]
    for item in spec:
        kind, name = item[0], item[1]
        if kind == "int":
            L.append(f"        int {name} = sc.nextInt();")
        elif kind == "pair":
            for x in name:
                L.append(f"        int {x} = sc.nextInt();")
        elif kind == "array":
            L += ["        int n = sc.nextInt();", f"        int[] {name} = new int[n];",
                  "        for (int i = 0; i < n; i++) {", f"            {name}[i] = sc.nextInt();", "        }"]
        elif kind == "word":
            L.append(f"        String {name} = sc.next();")
        elif kind == "line":
            L.append(f"        String {name} = sc.nextLine();")
    L += ["", "        // TODO: write your solution here"]
    arr = _array_name(spec)
    if out == "int":
        L += ["        int result = 0;", "", "        System.out.println(result);"]
    elif out == "text":
        L += ['        String result = "";', "", "        System.out.println(result);"]
    elif out == "arr":
        L += ["        for (int i = 0; i < n; i++) {", f'            System.out.print({arr}[i] + " ");',
              "        }", "        System.out.println();"]
    else:
        L += ["        for (int i = 1; i <= n; i++) {", "            System.out.println(i);", "        }"]
    L += ["    }", "}"]
    return "\n".join(L) + "\n"


def starter_js(spec, out):
    L = ["const lines = require('fs').readFileSync(0, 'utf8').split(/\\r?\\n/);"]
    pos = 0
    for item in spec:
        kind, name = item[0], item[1]
        if kind == "int":
            L.append(f"const {name} = parseInt(lines[{pos}]);")
            pos += 1
        elif kind == "pair":
            L.append(f"const [{', '.join(name)}] = lines[{pos}].trim().split(/\\s+/).map(Number);")
            pos += 1
        elif kind == "array":
            L.append(f"const n = parseInt(lines[{pos}]);")
            L.append(f"const {name} = lines[{pos + 1}].trim().split(/\\s+/).map(Number);")
            pos += 2
        elif kind == "word":
            L.append(f"const {name} = lines[{pos}].trim();")
            pos += 1
        elif kind == "line":
            L.append(f"const {name} = lines[{pos}];")
            pos += 1
    L += ["", "// TODO: write your solution here"]
    arr = _array_name(spec)
    if out == "int":
        L += ["let result = 0;", "", "console.log(result);"]
    elif out == "text":
        L += ['let result = "";', "", "console.log(result);"]
    elif out == "arr":
        L += [f"let result = {arr}.slice();", "", 'console.log(result.join(" "));']
    else:
        L += ["for (let i = 1; i <= n; i++) {", "  console.log(i);", "}"]
    return "\n".join(L) + "\n"


STARTERS = {71: starter_python, 50: starter_c, 54: starter_cpp, 62: starter_java, 63: starter_js}


def build_public_problems() -> List[Dict[str, Any]]:
    """What /problems returns: no hidden tests, but starter code for all five languages."""
    public = []
    for p in PROBLEMS:
        public.append({
            "id": p["id"],
            "title": p["title"],
            "difficulty": p["difficulty"],
            "tags": p["tags"],
            "description": p["description"],
            "input_format": p["input_format"],
            "output_format": p["output_format"],
            "constraints": p["constraints"],
            "examples": p["examples"],
            "starter_code": {str(lid): make(p["spec"], p["out"]) for lid, make in STARTERS.items()},
        })
    return public


PUBLIC_PROBLEMS = build_public_problems()
PROBLEMS_BY_ID = {p["id"]: p for p in PROBLEMS}


def _sanity_check_problem_bank():
    ids = [p["id"] for p in PROBLEMS]
    assert len(ids) == len(set(ids)), "duplicate problem id"
    for p in PROBLEMS:
        assert len(p["hidden"]) >= 3, f"problem {p['id']} needs at least 3 hidden tests"
        assert len(p["examples"]) + len(p["hidden"]) <= 20, "too many tests for one Judge0 batch"


_sanity_check_problem_bank()


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def b64e(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def b64d(text: Optional[str]) -> str:
    if not text:
        return ""
    try:
        return base64.b64decode(text).decode("utf-8", "replace")
    except Exception:
        return ""


def clip(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"
    return text


def normalize_output(text: str) -> str:
    """Ignore trailing spaces on each line and blank lines at the end (like most judges)."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def verdict_for(status_id: int) -> str:
    return VERDICTS.get(status_id, "Internal Error")


# ----------------------------------------------------------------------------
# Very small per-IP rate limiter (protects the free Judge0 server from spam)
# ----------------------------------------------------------------------------
_hits: Dict[str, deque] = {}


def check_rate_limit(request: Request) -> None:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    if len(_hits) > 5000:
        _hits.clear()
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="You are sending requests too quickly. Please wait a few seconds and try again.")
    q.append(now)


# ----------------------------------------------------------------------------
# Judge0 client
# ----------------------------------------------------------------------------
class Judge0Rejected(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(body)
        self.status = status
        self.body = body


JOB_SLOTS = asyncio.Semaphore(MAX_PARALLEL_JUDGE0_JOBS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if JUDGE0_AUTH_TOKEN:
        headers["X-Auth-Token"] = JUDGE0_AUTH_TOKEN
    app.state.http = httpx.AsyncClient(base_url=JUDGE0_URL, headers=headers, timeout=20.0)
    yield
    await app.state.http.aclose()


async def judge0(client: httpx.AsyncClient, method: str, path: str, *, params=None, body=None, timeout=20.0):
    """One HTTP call to Judge0 with friendly errors."""
    try:
        resp = await client.request(method, path, params=params, json=body, timeout=timeout)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="The code-execution server (Judge0) took too long to answer. Please try again.")
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Could not reach the code-execution server (Judge0). Please try again in a moment.")
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="The free Judge0 server is busy (rate limit). Wait about 10 seconds and try again.")
    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail=f"Judge0 had an internal problem (HTTP {resp.status_code}). Please try again later.")
    if resp.status_code >= 400:
        raise Judge0Rejected(resp.status_code, resp.text[:500])
    try:
        return resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Judge0 returned an unreadable response.")


def make_payload(language_id: int, source_code: str, stdin: str, with_limits: bool = True) -> Dict[str, Any]:
    lang = LANGUAGES[language_id]
    payload: Dict[str, Any] = {"language_id": language_id, "source_code": b64e(source_code)}
    if stdin:
        payload["stdin"] = b64e(stdin)
    if with_limits:
        payload["cpu_time_limit"] = lang["cpu"]
        payload["wall_time_limit"] = lang["wall"]
        payload["memory_limit"] = lang["memory"]
    return payload


async def create_submissions(client, language_id: int, source_code: str, stdins: List[str]) -> List[str]:
    """Send one batch to Judge0 and return the tokens (one per stdin)."""
    payloads = [make_payload(language_id, source_code, s) for s in stdins]
    try:
        data = await judge0(client, "POST", "/submissions/batch", params={"base64_encoded": "true"},
                            body={"submissions": payloads})
    except Judge0Rejected as err:
        if err.status == 422:
            # Probably a limit above what this Judge0 allows: retry once with Judge0's default limits.
            payloads = [make_payload(language_id, source_code, s, with_limits=False) for s in stdins]
            try:
                data = await judge0(client, "POST", "/submissions/batch", params={"base64_encoded": "true"},
                                    body={"submissions": payloads})
            except Judge0Rejected as err2:
                raise HTTPException(status_code=502, detail=f"Judge0 rejected the submission: {err2.body}")
        else:
            raise HTTPException(status_code=502, detail=f"Judge0 rejected the submission (HTTP {err.status}).")
    tokens = [item.get("token") if isinstance(item, dict) else None for item in (data or [])]
    if len(tokens) != len(stdins) or not all(tokens):
        raise HTTPException(status_code=502, detail="Judge0 did not return submission tokens.")
    return tokens


async def wait_for_results(client, tokens: List[str]) -> List[Dict[str, Any]]:
    """Poll Judge0 until every submission has finished (status id > 2)."""
    started = time.monotonic()
    delay = 0.6
    fields = "token,stdout,stderr,compile_output,message,status,time,memory"
    while True:
        data = await judge0(client, "GET", "/submissions/batch",
                            params={"tokens": ",".join(tokens), "base64_encoded": "true", "fields": fields})
        subs = (data or {}).get("submissions") or []
        by_token = {s.get("token"): s for s in subs if isinstance(s, dict)}
        ordered = [by_token.get(t) for t in tokens]
        if all(s and (s.get("status") or {}).get("id", 0) > 2 for s in ordered):
            return ordered
        if time.monotonic() - started > POLL_DEADLINE_SECONDS:
            raise HTTPException(status_code=504, detail="Execution did not finish in time. Judge0 may be overloaded; please try again.")
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, 2.0)


def describe(sub: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one raw Judge0 result into a clean dict for the frontend."""
    status = sub.get("status") or {}
    status_id = int(status.get("id", 13))
    compile_output = clip(b64d(sub.get("compile_output")))
    hint = ""
    if "should be declared in a file named" in compile_output:
        hint = "Java tip: the public class must be named Main (public class Main)."
    mem = sub.get("memory")
    return {
        "status_id": status_id,
        "status_description": status.get("description", "Unknown"),
        "verdict": verdict_for(status_id),
        "stdout": clip(b64d(sub.get("stdout"))),
        "stderr": clip(b64d(sub.get("stderr"))),
        "compile_output": compile_output,
        "message": clip(b64d(sub.get("message"))),
        "time": sub.get("time"),
        "memory": mem,
        "hint": hint,
    }


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="CodeForge API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://codeforge-[a-z0-9-]+\.vercel\.app",  # Vercel preview URLs
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    language_id: int = 71
    source_code: Optional[str] = None
    stdin: Optional[str] = ""
    # Old field names from the first version of CodeForge are still accepted:
    code: Optional[str] = None
    input: Optional[str] = None


class SubmitRequest(BaseModel):
    problem_id: int
    language_id: int = 71
    source_code: Optional[str] = None
    code: Optional[str] = None  # old field name


def validate_code(language_id: int, code: Optional[str]) -> str:
    if language_id not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language_id {language_id}. Use one of {sorted(LANGUAGES)}.")
    if code is None or not code.strip():
        raise HTTPException(status_code=400, detail="Your code is empty. Write some code first.")
    if len(code) > MAX_CODE_CHARS:
        raise HTTPException(status_code=413, detail=f"Code is too long (limit {MAX_CODE_CHARS} characters).")
    return code


@app.get("/")
def home():
    return {
        "message": "Welcome to CodeForge!",
        "status": "ok",
        "problems": len(PROBLEMS),
        "languages": [{"id": k, "name": v["name"]} for k, v in LANGUAGES.items()],
        "judge0": JUDGE0_URL,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/judge0-check")
async def judge0_check(request: Request):
    """Open this URL in the browser to see whether Judge0 is reachable from the server."""
    client = request.app.state.http
    try:
        info = await judge0(client, "GET", "/about", timeout=10.0)
    except Judge0Rejected as err:
        return {"judge0_reachable": False, "detail": f"HTTP {err.status}"}
    except HTTPException as err:
        return {"judge0_reachable": False, "detail": err.detail}
    return {"judge0_reachable": True, "judge0_url": JUDGE0_URL, "about": info}


@app.get("/languages")
def languages():
    return [{"id": k, "name": v["name"], "monaco": v["monaco"]} for k, v in LANGUAGES.items()]


@app.get("/problems")
def problems():
    return PUBLIC_PROBLEMS


@app.post("/run")
async def run_code(body: RunRequest, request: Request):
    check_rate_limit(request)
    code = validate_code(body.language_id, body.source_code if body.source_code is not None else body.code)
    stdin = body.stdin if body.stdin else (body.input or "")
    if len(stdin) > MAX_STDIN_CHARS:
        raise HTTPException(status_code=413, detail=f"Input is too long (limit {MAX_STDIN_CHARS} characters).")

    async with JOB_SLOTS:
        client = request.app.state.http
        tokens = await create_submissions(client, body.language_id, code, [stdin])
        results = await wait_for_results(client, tokens)

    info = describe(results[0])
    if info["status_id"] == 3:
        info["verdict"] = "Success"  # the program ran fine; /run does not judge the answer
    return info


@app.post("/submit")
async def submit_code(body: SubmitRequest, request: Request):
    check_rate_limit(request)
    code = validate_code(body.language_id, body.source_code if body.source_code is not None else body.code)
    problem = PROBLEMS_BY_ID.get(body.problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail=f"Problem {body.problem_id} does not exist.")

    # Samples first (their details are shown to the student), then the hidden tests.
    tests = [((ex["input"], ex["output"]), True) for ex in problem["examples"]] + \
            [((h[0], h[1]), False) for h in problem["hidden"]]
    stdins = [case[0] for case, _ in tests]

    async with JOB_SLOTS:
        client = request.app.state.http
        tokens = await create_submissions(client, body.language_id, code, stdins)
        raw_results = await wait_for_results(client, tokens)

    rows = []
    passed = 0
    first_failure: Optional[Dict[str, Any]] = None
    compile_output = ""
    max_time = 0.0
    max_memory = 0

    for number, ((case, is_sample), raw) in enumerate(zip(tests, raw_results), start=1):
        info = describe(raw)
        stdin, expected = case
        ok = False
        verdict = info["verdict"]
        if info["status_id"] == 3:
            ok = normalize_output(info["stdout"]) == normalize_output(expected)
            verdict = "Accepted" if ok else "Wrong Answer"
        if info["status_id"] == 6 and not compile_output:
            compile_output = info["compile_output"]
        try:
            max_time = max(max_time, float(info["time"] or 0))
        except (TypeError, ValueError):
            pass
        max_memory = max(max_memory, int(info["memory"] or 0))

        row = {
            "index": number,
            "sample": is_sample,
            "passed": ok,
            "verdict": verdict,
            "time": info["time"],
            "memory": info["memory"],
        }
        if is_sample:  # hidden tests never reveal their data
            row["input"] = stdin
            row["expected_output"] = expected
            row["actual_output"] = info["stdout"]
        if not ok and info["stderr"]:
            row["stderr"] = info["stderr"]
        rows.append(row)
        if ok:
            passed += 1
        elif first_failure is None:
            first_failure = {"verdict": verdict, "info": info, "number": number}

    total = len(rows)
    if any(r["verdict"] == "Compilation Error" for r in rows):
        verdict = "Compilation Error"
    elif first_failure is not None:
        verdict = first_failure["verdict"]
    else:
        verdict = "Accepted"

    response: Dict[str, Any] = {
        "problem_id": problem["id"],
        "language_id": body.language_id,
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "results": rows,
        "max_time": round(max_time, 3),
        "max_memory": max_memory,
        "compile_output": compile_output,
        "stderr": "",
        "message": "",
        "hint": "",
    }
    if first_failure is not None:
        fi = first_failure["info"]
        response["stderr"] = fi["stderr"]
        response["message"] = fi["message"]
        response["hint"] = fi["hint"]
        response["failed_test"] = first_failure["number"]
    return response
