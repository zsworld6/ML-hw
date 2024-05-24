from typing import Callable

from executor.py_executor import PyExecutor
from generator.py_generate import PyGenerator
from generator.model import llama3Model
from utility import get_prompts

from typing import List

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_reflexion(
    chat: Callable[[str], str],
    max_iters: int,
    data_path: str,
) -> None:
    exe = PyExecutor(data_path)
    gen = PyGenerator()
    
    is_solved = False
    solution = []
    
    implementations = []
    cur_func_impl = ""
    
    prompt = get_prompts(data_path)
    
    print("Initial Attempt")
    
    cur_func_impl = gen.func_impl(prompt, chat, "simple")
    implementations.append(cur_func_impl)
    is_passing, feedback = exe.evaluate(cur_func_impl)

    print("Initial Attempt Evaluate")
    # if solved, exit early
    if is_passing:
        is_solved = True
        solution.append(cur_func_impl, feedback)

    # input()

    # use self-reflection to iteratively improve
    cur_iter = 1
    cur_feedback = feedback
    while cur_iter < max_iters:
        print(f"Begin Attempt {cur_iter}")
        
        print("Get Self-reflection")
        reflection = gen.self_reflection(cur_func_impl, cur_feedback, chat)

        print("Apply Self-reflection in the next attempt")
        cur_func_impl = gen.func_impl(
            func_sig=prompt,
            chat=chat,
            strategy="reflexion",
            prev_func_impl=cur_func_impl,
            feedback=cur_feedback,
            self_reflection=reflection,
        )
        implementations.append(cur_func_impl)

        print("Evaluate")
        is_passing, cur_feedback = exe.evaluate(cur_func_impl)

        # if solved, check if it passes the real tests, exit early
        if is_passing:
            is_solved = True
            solution.append(cur_func_impl, cur_feedback)

        cur_iter += 1
        
        # input()

    if is_solved:
        with open("log", "w") as f:
            f.write(solution)
    else:
        print("Fail\n")
