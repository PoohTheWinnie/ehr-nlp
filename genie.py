import json

import pandas as pd
from vllm import LLM, SamplingParams

def process_ehr_data(
    model_path="/n/data1/hsph/biostat/celehs/lab/hongyi/ehrllm/THUMedInfo/GENIE_en_8b",
    data_path="data/mimic_smoking.csv",
    n_samples=2,
    temperature=0.7,
    max_new_token=512,
    tensor_parallel_size=1,
):
    """
    Process EHR data using the specified LLM model.

    Args:
        model_path (str): Path to the model
        data_path (str): Path to the CSV file containing EHR data
        n_samples (int): Number of samples to process
        temperature (float): Sampling temperature
        max_new_token (int): Maximum number of tokens to generate
        tensor_parallel_size (int): Tensor parallel size for model

    Returns:
        list: List of processed responses
    """
    PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"

    model = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token)

    # Load and prepare data
    df = pd.read_csv(data_path)
    first_n_texts = df["text"].head(n_samples)
    EHR = first_n_texts.tolist()

    # Generate prompts and get model outputs
    texts = [PROMPT_TEMPLATE.format(query=k) for k in EHR]
    outputs = model.generate(texts, sampling_params)

    # Process responses
    responses = []
    for output in outputs:
        response = output.outputs[0].text
        last_closed_brace_index = response.rfind('}')
        complete_json_string = response[:last_closed_brace_index + 1] + ']'
        responses.append(json.loads(complete_json_string))

    return responses


if __name__ == "__main__":
    responses = process_ehr_data()
    print(responses)
    # res = json.loads(output[0])


