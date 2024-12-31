import json
import re

import pandas as pd

file = "../local/mimic_smoking.csv"
dataframe = pd.read_csv(file)

raw_contexts = []
for index, row in dataframe.iterrows():
    if index == 100:
        break
    raw_contexts.append(row["text"])
    # text = row["text"]
    # sentences = re.findall(r'([^.]*?smoking[^.]*\.)', text)
    # sentences = [sentence.lstrip() for sentence in sentences]
    # raw_contexts.extend(sentences)

raw_questions = [
    "Given the fact that this is a patient's doctor's note, is this patient a current smoker, past smoker, or has never smoked before?"
] * len(raw_contexts)
with open("../celehs_llm_scripts/inference/dummy_data.jsonl", "w") as f:
    for idx, item in enumerate(raw_questions):
        f.write(
            json.dumps(
                {
                    "sample_idx": idx,
                    "context": raw_contexts[idx],
                    "question": item,
                    "answer": "I don't know.",
                }
            )
            + "\n"
        )
