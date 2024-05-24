from typing import List, Union, Optional, Literal
import dataclasses

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

class ModelBase():
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 1) -> Union[List[str], str]:
        raise NotImplementedError

class llama3Model(ModelBase):
    def __init__():
        super.__init__("llama3")
        
    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 1) -> Union[List[str], str]:
        # TODO: try to call llama
        pass