import argparse
import os
from typing import List, Tuple

import pandas as pd
import torch
from fastchat.model import get_conversation_template
from fastchat.model.model_adapter import register_model_adapter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import CancerData, SmokingData
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


class ModelInference:
    def __init__(self, model_path: str, model_id: str, max_new_token: int = 1024):
        """
        Initializes the ModelInference class with model and tokenizer.

        Args:
            model_path (str): Path or ID of the model to load.
            model_id (str): Identifier for the model.
            max_new_token (int): Maximum number of tokens to generate.
        """
        self.model_path = model_path
        self.model_id = model_id
        self.max_new_token = max_new_token

        # Set device to MPS if available, else CPU
        self.device = torch.device("cpu")

        # Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, padding_side="right"
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self._initialize_special_tokens()

    def _initialize_special_tokens(self) -> None:
        """
        Initializes special tokens if they are missing in the tokenizer.
        """
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

    def generate_outputs(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[List[str], List[str]]:
        """
        Generates outputs from the model based on input dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset containing contexts and questions.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing a list of generated outputs and the list of prompts.
        """
        data_loader = DataLoader(dataset, batch_size=1)
        prompts = []
        outputs = []

        for item in tqdm(data_loader, desc="Prompt Initialization"):
            question_prompt = few_shot_question_template.format(
                context=item["context"][0], question=item["question"][0]
            )
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], question_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

            # Generate output
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.device
            )
            output_ids = self.model.generate(input_ids, max_length=self.max_new_token)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output_text)

        return outputs, prompts

    def extract_embeddings(
        self, outputs: List[str], prompts: List[str], embedding_type: str = "Average"
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Extracts embeddings from model outputs and prompts.

        Args:
            outputs (List[str]): List of generated outputs from the model.
            prompts (List[str]): List of prompts corresponding to each output.
            embedding_type (str): Type of embedding extraction ('Head' or 'Average').

        Returns:
            Tuple[List[List[float]], List[List[float]]]: A tuple containing lists of input and output embeddings.
        """
        output_embeddings, input_embeddings = [], []

        for output, prompt in tqdm(zip(outputs, prompts), desc="Extracting embeddings"):
            with torch.no_grad():
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
                output_ids = self.tokenizer(output, return_tensors="pt").input_ids.to(
                    self.device
                )

                input_embedding = self._compute_embedding(input_ids, embedding_type)
                output_embedding = self._compute_embedding(output_ids, embedding_type)

                input_embeddings.append(input_embedding)
                output_embeddings.append(output_embedding)

        return input_embeddings, output_embeddings

    def _compute_embedding(
        self, token_ids: torch.Tensor, embedding_type: str
    ) -> List[float]:
        """
        Computes embeddings for a given tensor of token IDs.

        Args:
            token_ids (torch.Tensor): The token IDs for which to compute embeddings.
            embedding_type (str): Type of embedding extraction ('Head' or 'Average').

        Returns:
            List[float]: Computed embeddings as a list of floats.
        """
        model_output = self.model(
            token_ids, return_dict=True, output_hidden_states=True
        )
        hidden_state = model_output.hidden_states[-1]

        if embedding_type == "Head":
            return hidden_state[0, 0, :].tolist()
        elif embedding_type == "Average":
            average_embedding = torch.mean(hidden_state, dim=1)
            return average_embedding.view(-1).tolist()

    def save_embeddings(self, embeddings: List[List[float]], file_path: str) -> None:
        """
        Saves embeddings to a specified CSV file.

        Args:
            embeddings (List[List[float]]): List of embeddings to save.
            file_path (str): Path to the output CSV file.
        """
        df = pd.DataFrame(embeddings).T
        df.to_csv(file_path, sep="\t", encoding="utf-8", index=False)


def main(args: argparse.Namespace) -> None:
    """
    Main function to run model inference and extract embeddings.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
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
