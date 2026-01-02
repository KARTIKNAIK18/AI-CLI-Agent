import os
from openai import OpenAI
import json
from dotenv import load_dotenv


load_dotenv('../.env')

# model_name = "qwen3-vl:2b"
model_name = "openai/gpt-5"
# base_url = "http://127.0.0.1:11434/v1"
base_url = "https://models.github.ai/inference"
client = OpenAI(
    base_url=base_url,
    api_key=os.getenv('GITHUB_TOKEN')
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
        "steps": "TOOL",
        "content": "To use the ls command, I will use the run_cmnd tool."
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

def run_cmnd(cmnd: str):
    result = os.system(cmnd)
    return result

message_hist = [
    {"role": "system", "content": SYSTEM_MESSAGE }
]


avilable_tools = {
    "run_cmnd": run_cmnd
}

user_input  = input("Enter your query: ")

while True:
    try:
        response = client.chat.completions.create(
            model = model_name,
            response_format={"type": "json_object"},
            messages=message_hist   
        )

        raw_response = response.choices[0].message.content
        try:
            output = json.loads(raw_response)
            print("AI Response:", output)

            message_hist.append({"role": "assistant", "content": raw_response})
        except json.JSONDecodeError:
            print("Failed to parse JSON response:", raw_response)

        if output.get("steps") == "START":
            print("Starting:", output.get("content"))
            message_hist.append({"role": "assistant", "content": json.dumps(output)})
            continue
        if output.get("steps") == "TOOL":
            tool_to_call = output.get("tool")
            tool_input = output.get("input")
            print(f"🔧 TOOL: {tool_to_call} with input {tool_input}")

            tool_res = avilable_tools[tool_to_call](tool_input)
            print(f"🛠️ TOOL RESULT: {tool_res}")
        if output.get("steps") == "PLAN":
            print("Planning:", output.get("content"))
            message_hist.append({"role": "assistant", "content": json.dumps(output)})
            continue

        if output.get("steps") == "SOLVE":
            print("Final Output:", output.get("content"))

        if output.get("steps") == "OUTPUT":
            print("Final Output:", output.get("content"))
            break
    except Exception as e:
        print("Error:", str(e))
        break
        









