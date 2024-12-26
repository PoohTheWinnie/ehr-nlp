import json


raw_questions = [
    "Who is this context talking about",
    "where does he work",
    "what is his job position?",
]

raw_contexts = [
    "There is a person named john working at Harvard Medical School. He works as a faculty staff",
    "There is a person named john working at Harvard Medical School. He works as a faculty staff",
    "There is a person named john working at Harvard Medical School. He works as a faculty staff",
]


with open("./dummy_data.jsonl", "w") as f:
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
