import argparse
import json
import os
import re
import time

import pandas as pd
import torch
from fastchat.model import get_conversation_template
from fastchat.model.model_adapter import register_model_adapter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from templates import (
    EeveeAdapter,
    FewShotAdapter,
    PretrainFewShotAdapter,
    few_shot_question_template,
)

# Register adapters
register_model_adapter(EeveeAdapter)
register_model_adapter(FewShotAdapter)
register_model_adapter(PretrainFewShotAdapter)


class SmokingData(Dataset):
    def __init__(self, file_path, max_samples=100):
        self.data = pd.read_csv(file_path).head(max_samples)
        self.contexts = self.data["text"].tolist()
        self.questions = [
            "Given the fact that the following is an excerpt from a patient's doctor's note, "
            "is this patient a current smoker, past smoker, or has never smoked before?"
        ] * len(self.contexts)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {"context": self.contexts[idx], "question": self.questions[idx]}


class CancerData(Dataset):
    def __init__(self, file_path, max_samples=100):
        self.data = pd.read_csv(file_path).head(max_samples)
        self.contexts = []
        for text in self.data["text"]:
            sentences = re.findall(r"([^.]*?cancer[^.]*\.)", text)
            self.contexts.extend([sentence.lstrip() for sentence in sentences])
        self.questions = [
            "Given the fact that the following is an excerpt from a patient's doctor's note, "
            "what is the stage of cancer does the patient have?"
        ] * len(self.contexts)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {"context": self.contexts[idx], "question": self.questions[idx]}


class ModelInference:
    def __init__(self, model_path, model_id, tp_size=1, max_new_token=1024):
        self.model_path = model_path
        self.model_id = model_id
        self.tp_size = tp_size
        self.max_new_token = max_new_token

        # Set device to MPS if available for MAC
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, padding_side="right"
        )
        self._initialize_special_tokens()

        # Load Model
        self.model = LLM(
            model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_token)

    def _initialize_special_tokens(self):
        special_tokens_dict = {
            k: v
            for k, v in {
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }.items()
            if getattr(self.tokenizer, k) is None
        }
        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.tokenizer.save_pretrained(self.model_path)

    def generate_outputs(self, dataset):
        data_loader = DataLoader(dataset, batch_size=1)
        prompts, inputs = [], []
        for item in tqdm(data_loader, desc="Prompt Initialization"):
            question_prompt = few_shot_question_template.format(
                context=item["context"][0], question=item["question"][0]
            )
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], question_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            inputs.append(self.tokenizer(prompt, return_tensors="pt"))

        # Generate output
        outputs = self.model.generate(prompts, self.sampling_params)
        return outputs, prompts

    def extract_embeddings(self, outputs, prompts, embedding_type="Average"):
        output_embeddings, input_embeddings = [], []
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(self.device)

        for output, prompt in tqdm(zip(outputs, prompts), desc="Extracting embeddings"):
            with torch.no_grad():
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
                output_ids = output.outputs[0].token_ids
                question_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids

                input_embedding = self._compute_embedding(
                    model, input_ids, embedding_type
                )
                output_embedding = self._compute_embedding(
                    model, output_ids, embedding_type
                )

                input_embeddings.append(input_embedding)
                output_embeddings.append(output_embedding)

        return input_embeddings, output_embeddings

    def _compute_embedding(self, model, token_ids, embedding_type):
        model_output = model(token_ids, return_dict=True, output_hidden_states=True)
        hidden_state = model_output.hidden_states[-1]

        if embedding_type == "Head":
            return hidden_state[0, 0, :].tolist()
        elif embedding_type == "Average":
            average_embedding = torch.mean(hidden_state, dim=1)
            return average_embedding.view(-1).tolist()

    def save_embeddings(self, embeddings, file_path):
        df = pd.DataFrame(embeddings).T
        df.to_csv(file_path, sep="\t", encoding="utf-8", index=False)


def main(args):
    # Select dataset
    if args.dataset == "smoking":
        dataset = SmokingData(file_path=args.data_file)
    elif args.dataset == "cancer":
        dataset = CancerData(file_path=args.data_file)
    else:
        raise ValueError("Invalid dataset choice. Use 'smoking' or 'cancer'.")

    # Initialize ModelInference
    model_inference = ModelInference(
        model_path=args.model_path,
        model_id=args.model_id,
        tp_size=args.tp_size,
        max_new_token=args.max_new_token,
    )

    # Generate and save outputs
    outputs, prompts = model_inference.generate_outputs(dataset)
    input_embeddings, output_embeddings = model_inference.extract_embeddings(
        outputs, prompts, embedding_type="Average"
    )
    model_inference.save_embeddings(
        input_embeddings, os.path.join(args.output_dir, "input_embeddings.csv")
    )
    model_inference.save_embeddings(
        output_embeddings, os.path.join(args.output_dir, "output_embeddings.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model weights."
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to the dataset file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["smoking", "cancer"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save outputs."
    )
    parser.add_argument(
        "--max-new-token", type=int, default=1024, help="Max tokens generated."
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")

    args = parser.parse_args()
    main(args)
