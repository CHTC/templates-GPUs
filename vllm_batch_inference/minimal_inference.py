import json
from pathlib import Path

import vllm

MODEL = "microsoft/Phi-3.5-mini-instruct"

inputs = [
    "What is your favorite color?",
    "How old are you?",
    "Do you have a pet?",
    "What is your favorite fruit?",
    "What do you like to do for fun?",
]


def format_prompt(user_message: str) -> str:
    """Each model has a specific prompt format that it expects. This function formats the user message into the correct format for the `Phi-3.5-mini-instruct` model.
    See: https://huggingface.co/microsoft/Phi-3.5-mini-instruct#input-formats
    """
    return (
        "<|system|>\n"
        "You are a helpful assistant. Provide short answers.<|end|>\n"
        "<|user|>\n"
        f"{user_message}<|end|>\n"
        "<|assistant|>\n"
    )


def inference(inputs: list[str], model: str = MODEL) -> list[str]:
    """Perform batch inference."""
    formatted_inputs = [format_prompt(user_message=x) for x in inputs]
    llm = vllm.LLM(model=model)
    outputs = llm.generate(prompts=formatted_inputs)
    return [raw_output.outputs[0].text.strip() for raw_output in outputs]  # text only


def save_data(inputs: list[str], outputs: list[str], file: Path) -> None:
    """Save data to a jsonl file."""
    with open(file, mode="a") as f:
        for x, y in zip(inputs, outputs):
            data = {"input": x, "output": y}
            f.write(json.dumps(data) + "\n")


outputs = inference(inputs)
save_data(inputs, outputs, Path("outputs.jsonl"))
