from .model import Message
import dataclasses

from typing import Union, List, Optional, Callable, Dict

def generic_generate_func_impl(
    func_sig: str,
    chat: Callable[[List[Dict]], str],
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    reflexion_chat_instruction: str,
    simple_chat_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> Union[str, List[str]]:
    if strategy == "reflexion":
        # message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
        prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}"
        # func_bodies is a really bad name, as it can also be just 1 string
        # print_messages(prompt, message)
        messages = [
            Message(
                role="system",
                content=prompt,
            ),
            Message(
                role="assistant",
                content=add_code_block(prev_func_impl),
            ),
            Message(
                role="user",
                content=f"[feedback from previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
            ),
            Message(
                role="assistant",
                content=self_reflection,
            ),
            Message(
                role="user",
                # content=f"[improved impl]:\n{func_sig}",
                content=f"[improved impl]:\n",
            ),
        ]
        print("--------------------------------------------------------------------------------")
        print(messages)
        print("--------------------------------------------------------------------------------")
        func_bodies = chat([dataclasses.asdict(message) for message in messages])
    else:
        system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
        print_messages(system_prompt, func_sig)
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
        func_bodies = chat([dataclasses.asdict(message) for message in messages])

    func_body_str = parse_code_block(func_bodies)
    print_generated_func_body(func_body_str)
    return func_body_str


def generic_generate_self_reflection(
        func: str,
        feedback: str,
        chat: Callable[[List[Dict]], str],
        self_reflection_chat_instruction: str,
        add_code_block: Callable[[str], str],
) -> str:
    print(f'Feedback output: {feedback}')
    messages = [
        Message(
            role="system",
            content=self_reflection_chat_instruction,
        ),
        Message(
            role="user",
            content=f'[function impl]:\n{add_code_block(func)}\n\n[feedback]:\n{feedback}\n\n[self-reflection]:',
        )
    ]
    reflection = chat([dataclasses.asdict(message) for message in messages])
    print(f'Self reflection output: {reflection}')
    return reflection  # type: ignore


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
