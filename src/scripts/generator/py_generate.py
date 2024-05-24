from typing import Optional, List, Union, Callable, Dict

from .generator_utils import generic_generate_func_impl, generic_generate_self_reflection
from .parse import parse_code_block, add_code_block

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT CHINESE, NOT ENGLISH. You will be given a task by the user. Write your full implementation."
# PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only python code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past implementation, a feedback of running your code, and a hint to change the implementation appropriately. Write your full implementation."
# PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation (restate the function signature)."

PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Python programming assistant. You will be given a implementation and a feedback of running your code. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the feedback. If your implementation is correct, try to improve it. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
# PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Python programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."

class PyGenerator:
    def self_reflection(self, func: str, feedback: str, chat: Callable[[List[Dict]], str]) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            chat=chat,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "python"),
        )

    def func_impl(
        self,
        func_sig: str,
        chat: Callable[[List[Dict]], str],
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
    ) -> Union[str, List[str]]:
        return generic_generate_func_impl(
            func_sig=func_sig,
            chat=chat,
            strategy=strategy,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            reflexion_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "python"),
            add_code_block=lambda x: add_code_block(x, "python"),
        )

