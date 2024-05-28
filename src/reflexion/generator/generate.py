from typing import Optional, List, Union, Callable, Dict
from .utils import Message, parse_code_block, add_code_block
import dataclasses
import argparse
from .model import get_model_from_path

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"

PY_SIMPLE_INSTRUCTION = "You are an AI that only responds with python code, NOT CHINESE, NOT ENGLISH. You will be given a task by the user. Write your full implementation."
# PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only python code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
PY_REFLEXION_INSTRUCTION = "You are an AI Python assistant. You will be given your past implementation, some error messages of your code, and a hint to change the implementation appropriately. Write your full implementation."
# PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation (restate the function signature)."

PY_SELF_REFLECTION_INSTRUCTION = "You are a Python programming assistant. You will be given a implementation and some error messages of your code. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the errors. If your implementation is correct, try to improve it. You will need this as a hint when you try again later. Remember forever, only provide the few sentence description in your answer, not the implementation!"
# PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Python programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."

class PyGenerator:
    def __init__(self, args):
        self.model = get_model_from_path(args.model_path)
        self.args = args
    def self_reflection(self, func: str, feedback: str) -> str:
        print(f'Feedback output: {feedback}')
        messages = [
            Message(
                role="system",
                content=PY_SELF_REFLECTION_INSTRUCTION,
            ),
            Message(
                role="user",
                content=f'[impl]:\n{add_code_block(func)}\n\n[feedback]:\n{feedback}\n\n[self-reflection]:',
            )
        ]
        reflection = self.model.generate([dataclasses.asdict(message) for message in messages], self.args)
        print(f'Self reflection output: {reflection}')
        return reflection  # type: ignore
    

    def func_impl(
        self,
        func_sig: str,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
    ) -> Union[str, List[str]]:
        if strategy == "reflexion":
            prompt = f"{PY_REFLEXION_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}"
            # func_bodies is a really bad name, as it can also be just 1 string
            # print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
                Message(
                    role="user",
                    content=func_sig,
                ),
                Message(
                    role="assistant",
                    content=add_code_block(prev_func_impl),
                ),
                Message(
                    role="user",
                    content=f"[error messages of previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
                ),
                Message(
                    role="assistant",
                    content=self_reflection,
                ),
                Message(
                    role="user",
                    content=f"[improved impl]:\n",
                    # content=f"[improved impl]:\n",
                ),
            ]
            # print("--------------------------------------------------------------------------------")
            # print(messages)
            # print("--------------------------------------------------------------------------------")
            func_bodies = self.model.generate([dataclasses.asdict(message) for message in messages], self.args)
        else:
            system_prompt = f"{PY_SIMPLE_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}"
            # print_messages(system_prompt, func_sig)
            messages = [
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=func_sig,
                ),
            ]
            func_bodies = self.model.generate([dataclasses.asdict(message) for message in messages], self.args)

        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str
    

def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)

def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")
