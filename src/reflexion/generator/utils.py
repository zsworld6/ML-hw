import re
import dataclasses
from typing import List, Optional, Literal

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])



def parse_code_block(string: str, lang: str = "python") -> Optional[str]:
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    # return string
    return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str = "python") -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    got_return = False
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                break
        elif "return" in line and def_i != -1:
            got_return = True
        if line == "" and def_i != -1 and got_return:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == -1:
        return None

    return "\n".join(code_lines[def_i:last_i+1]).rstrip("[/PYTHON]")


def add_code_block(string: str, lang: str = "python") -> str:
    return f"```{lang}\n{string}\n```"

