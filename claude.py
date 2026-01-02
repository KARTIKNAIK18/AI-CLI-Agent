import os
from openai import OpenAI
import json
from pydantic import BaseModel
from dotenv import load_dotenv



# model_name = "qwen3-vl:2b"
model_name = "openai/gpt-5"
# base_url = "http://127.0.0.1:11434/v1"
base_url = "https://models.github.ai/inference"
client = OpenAI(
    base_url=base_url,
    api_key=load_dotenv('../env')
)



SYSTEM_MESSAGE = """
    you're an expert ai Assistant that  master in  resovling user query using a chain of thought process.
    you work on 5 steps: START, PLAN, SOLVE, OBSERVE, OUTPUT.
    you  need to first plan how to solve the problem . think  what has to  be done  the plan can be multi steps.
    once you have prepared you're plan then you can go for the output,

    Rules:
    - only run one step at a time
    - start from the START step
    - seqence of steps are START (where user gives the input),
        PLAN( That can be mulitple times),
        SOLVE (that can be multiple times),
        OBSERVE (that is the output from the tool) and
        finally OUTPUT (which is going to display to the user)
    - output must be in json format as mentioned below.

    Tools:
    - run_cmnd(cmnd: str): runs a terminal command and returns the output.
    
     Output JSON Format:
    {
    "Steps" : "START" | "PLAN" | "OUTPUT" | "TOOL", "Content": "String", "tool": "String", "input": "String"
    }

    example:
    User Query: "List all files in the current directory"
    AI Response:
    {
        "steps": "START",
        "content": "User wants to list all files in the current directory."
    }
    {
        "steps": "PLAN",
        "content": "To list all files, I will use the 'ls' command in the terminal."
    }
    {
        "steps": "SOLVE",
        "tool": "run_cmnd",
        "input": "ls"
    }
    {
        "steps": "OBSERVE",
        "content": "The output of the 'ls' command is: file1.txt, file2.txt, script.py"
    }
    {
        "steps": "OUTPUT",
        "content": "The files in the current directory are: file1.txt, file2.txt, script.py"
    }
"""
user_input  = input("Enter your query: ")

def run_cmnd(cmnd: str):
    result = os.system(cmnd)
    return result

try:
    response = client.chat.completions.create(
        model = model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_input}
            
        ]    
    )

    raw_response = response.choices[0].message.content
    try:
        output = json.loads(raw_response)
        print("AI Response:", output)
    except json.JSONDecodeError:
        print("Failed to parse JSON response:", raw_response)
    if output.get("steps") == "OUTPUT":
        print("Final Output:", output.get("content"))

except Exception as e:
    print("Error:", str(e))
    









