import json
from pathlib import Path

import vllm


def load_data(file: Path) -> list[dict[str, str]]:
    """Load data from a jsonl file. Assuming at least `id` and `input` fields are present.

    Example input format: (docs: https://jsonlines.org/)
    {"id": "x0001", input": "What is the capital of France?"}
    {"id": "x0002", input": "What is the capital of Germany?"}
    """

    with open(file, mode="r") as f:
        return [json.loads(line) for line in f]


def format_prompt(user_message: str, system_message: str | None = None) -> str:
    """This function formats the user message into the correct format for the `Phi-3.5-mini-instruct` model.

    docs: https://huggingface.co/microsoft/Phi-3.5-mini-instruct#input-formats
    """

    if system_message is None:
        system_message = (
            "You are a helpful assistant. Provide clear and concise responses."
        )
    return (
        "<|system|>\n"
        f"{system_message}<|end|>\n"
        "<|user|>\n"
        f"{user_message}<|end|>\n"
        "<|assistant|>\n"
    )


def inference(data: list[dict[str, str]], input_field: str = "input") -> list[dict]:
    """Perform a batch of inference on Phi-3.5-mini-instruct via vllm offline mode.

    docs: https://docs.vllm.ai/en/v0.6.4/getting_started/examples/offline_inference.html
    """

    inputs = [item[input_field] for item in data]
    formatted_inputs = [format_prompt(user_message=x) for x in inputs]

    llm = vllm.LLM(model="microsoft/Phi-3.5-mini-instruct")

    # Set sampling parameters
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=2048)
    outputs = llm.generate(prompts=formatted_inputs, sampling_params=sampling_params)
    text_outputs = [raw_output.outputs[0].text.strip() for raw_output in outputs]
    return [{**item, "output": output} for item, output in zip(data, text_outputs)]


def save_data(data: list[dict], file: Path) -> None:
    """Save data to a jsonl file in append mode.

    Example output format:
    {"id": "x0001", "input": "What is the capital of France?", "output": "Paris."}
    {"id": "x0002", "input": "What is the capital of Germany?", "output": "Berlin."}
    """
    with open(file, mode="a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# Load data, perform inference, and save data
data = load_data(Path("inputs.jsonl"))
outputs = inference(data)
save_data(outputs, Path("outputs.jsonl"))
