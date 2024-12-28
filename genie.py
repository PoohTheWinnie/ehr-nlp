
import pandas as pd
from vllm import LLM, SamplingParams
import json

# Define parameters
temperature = 0.7  # Default temperature value
max_new_token = 512  # Default max tokens
model_path = 'THUMedInfo/GENIE_en_8b'
tensor_parallel_size = 1
PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"
date_path = 'ehr-nlp/data/mimic_smoking.csv'
n_samples = 100

model = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token)

df = pd.read_csv(date_path)
first_100_texts = df['text'].head(n_samples)
EHR = first_100_texts.tolist()

texts = [PROMPT_TEMPLATE.format(query=k) for k in EHR]
outputs = model.generate(texts, sampling_params)
responses = []
for output in outputs:
    try:
        response = json.loads(output.outputs[0].text)
        responses.append(response)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from output: {output.outputs[0].text}")
