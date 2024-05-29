from typing import Callable
import argparse

from reflexion.executor.executor import PyExecutor
from reflexion.generator.generate import PyGenerator

from utils.utility import get_prompts

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_reflexion(
    args: argparse.Namespace,
) -> None:
    exe = PyExecutor()
    gen = PyGenerator(args)
    
    implementations = []
    solution = []
    cur_func_impl = ""    
    prompt = get_prompts(args.data_path, args.describe_task)
    
    print("Initial Attempt")
    
    cur_func_impl = gen.func_impl(prompt, "simple")
    implementations.append(cur_func_impl)
    is_passing, feedback = exe.evaluate(cur_func_impl)

    print("Initial Attempt Evaluate")
    # if solved, exit early
    if is_passing:
        solution.append((cur_func_impl, feedback))

    # input()

    # use self-reflection to iteratively improve
    cur_iter = 1
    cur_feedback = feedback
    while cur_iter < args.reflexion_iterations and not is_passing:
        print(f"Begin Attempt {cur_iter}")
        
        print("Get Self-reflection")
        reflection = gen.self_reflection(cur_func_impl, cur_feedback)

        print("Apply Self-reflection in the next attempt")
        cur_func_impl = gen.func_impl(
            func_sig=prompt,
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
            solution.append((cur_func_impl, cur_feedback))

        cur_iter += 1
        
        # input()

    if is_passing:
        with open("log", "w") as f:
            f.write(solution)
    else:
        print("Fail\n")
