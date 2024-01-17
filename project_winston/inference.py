import os
import re
import json
import time
import torch
import argparse
import transformers

import pandas as pd

from tqdm import tqdm
from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template
from fastchat.model.model_adapter import register_model_adapter
from templates import EeveeAdapter, FewShotAdapter, PretrainFewShotAdapter, few_shot_question_template

register_model_adapter(EeveeAdapter)
register_model_adapter(FewShotAdapter)
register_model_adapter(PretrainFewShotAdapter)

def create_smoking_data():
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

    raw_questions = ["Given the fact that the following is an excerpt from a patients doctor's note, is this patient a current smoker, past smoker, or has never smoked before?"] * len(raw_contexts)
    with open('../celehs_llm_scripts/inference/dummy_data.jsonl', 'w') as f:
        for idx, item in enumerate(raw_questions):
            f.write(json.dumps({'sample_idx': idx, 'context': raw_contexts[idx], 'question': item, 'answer': 'I don\'t know.'}) + '\n')

def create_cancer_data():
    file = "../local/mimic_cancer.csv"
    dataframe = pd.read_csv(file)

    raw_contexts = []
    for index, row in dataframe.iterrows():
        if index == 100:
            break
        text = row["text"]
        sentences = re.findall(r'([^.]*?cancer[^.]*\.)', text)
        sentences = [sentence.lstrip() for sentence in sentences]
        raw_contexts.extend(sentences)

    raw_questions = ["Given the fact that the following is an excerpt from a patients doctor's note, what is the stage of cancer does the patient have?"] * len(raw_contexts)
    with open('../celehs_llm_scripts/inference/dummy_data.jsonl', 'w') as f:
        for idx, item in enumerate(raw_questions):
            f.write(json.dumps({'sample_idx': idx, 'context': raw_contexts[idx], 'question': item, 'answer': 'I don\'t know.'}) + '\n')

def run(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    tp_size,
    embedding_type=None
):
    start_time = time.time()
    # ====== Establish tokenizer ======
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = '<pad>'
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = '</s>'
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = '<s>'
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = '<unk>'
    if len(special_tokens_dict) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.save_pretrained(model_path)

    # ====== Load model ======
    try:
        model = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=0.9)
    except RecursionError:
        model = LLM(model=model_path, tokenizer_mode='slow', tensor_parallel_size=tp_size, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_token)

    # ===== Configure input prompts ======
    inputs = []
    prompts = []
    for item in tqdm(questions, desc="Prompt Initialization: "):
        torch.manual_seed(0)
        if 'llama' in model_id.lower() and 'chat' not in model_id.lower():
            conv = get_conversation_template('pretrainfewshot')
        else:
            conv = get_conversation_template(model_id)
        qs = few_shot_question_template.format(context=item['context'], question=item["question"])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
        inputs.append(tokenizer(prompt, return_tensors="pt"))

    # ====== Run model ======    
    outputs = model.generate(prompts, sampling_params)
    prompt_id_map = {prompt: idx for idx, prompt in enumerate(prompts)}

    output_token_ids = []
    for output in tqdm(outputs, desc="Generate text output: "):
        output_ids = output.outputs[0].token_ids
        question = questions[prompt_id_map[output.prompt]]

        # be consistent with the template's stop_token_ids
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = model.get_tokenizer().decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]
        for special_token in model.get_tokenizer().special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        output_token_ids.append(tokenizer(output, return_tensors="pt").input_ids)

        question['output'] = output
        question['generator'] = model_id

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(question) + "\n")
    
    # ====== Extract embeddings ======  
    if embedding_type is None:
        end_time = time.time()
        print(f"Total runtime: {str(end_time-start_time)} seconds")
    
    device = torch.device('cuda')
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    
    input_embeddings = []
    output_embeddings = []
    
    with torch.no_grad():
        question_tokens = tokenizer(questions[0]["question"], return_tensors="pt").input_ids        
        for i, input in tqdm(enumerate(output_token_ids), desc="Extracting embeddings: "):
            input = input.to(device)
            model_output = model(input, return_dict=True, output_hidden_states=True)
            
            if embedding_type == "Head":
                output_embeddings.append(model_output.hidden_states[-1][0, 0, :].tolist())
            if embedding_type == "Average":
                average_embedding = torch.mean(model_output.hidden_states[-1], dim=1)
                flattened_embedding = average_embedding.view(-1)
                output_embeddings.append(flattened_embedding.tolist())
            
            context_tokens = tokenizer(questions[0]["context"], return_tensors="pt").input_ids
            context_tokens = context_tokens.to(device)
            model_output = model(context_tokens, return_dict=True, output_hidden_states=True)
            print(model_output.hidden_states[-1].size())
            if embedding_type == "Head":
                input_embeddings.append(model_output.hidden_states[-1][0, -len(question_tokens), :].tolist())
            if embedding_type == "Average":
                average_embedding = torch.mean(model_output.hidden_states[-1][0, -len(question_tokens):, :], dim=1)
                flattened_embedding = average_embedding.view(-1)
                input_embeddings.append(flattened_embedding.tolist())

        question_tokens = question_tokens.to(device)
        model_output = model(question_tokens, return_dict=True, output_hidden_states=True)
        
        if embedding_type == "Head":
            input_embeddings.append(model_output.hidden_states[-1][0, 0, :].tolist())
        if embedding_type == "Average":
            average_embedding = torch.mean(model_output.hidden_states[-1], dim=1)
            flattened_embedding = average_embedding.view(-1)
            input_embeddings.append(flattened_embedding.tolist())
        
    input_embeddings = pd.DataFrame(input_embeddings).T
    input_embeddings.to_csv(os.path.dirname(answer_file) + "/input_embeddings.csv", sep='\t', encoding='utf-8', index=False)

    output_embeddings = pd.DataFrame(output_embeddings).T
    output_embeddings.to_csv(os.path.dirname(answer_file) + "/output_embeddings.csv", sep='\t', encoding='utf-8', index=False)
    
    # ====== Runtime ======    
    end_time = time.time()
    print(f"Total runtime: {str(end_time-start_time)} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="The output answer file.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )

    args = parser.parse_args()

    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]
    
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        num_replicas = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        tp_size = torch.cuda.device_count() // num_replicas
        devices = ','.join([str(i) for i in range(rank*tp_size, (rank+1)*tp_size)])
        # torch.cuda.set_device(devices)
        total_size = len(questions)
        questions = questions[rank:total_size:num_replicas]
        args.answer_file = args.answer_file.replace(".jsonl", f"_{rank}.jsonl")

        print(f"RANK: {rank} | NUM_REPLICAS: {num_replicas} | devices : {devices}")
    else:
        tp_size = 1

    print(f"Num Questions: {len(questions)}")
    
    run(
        args.model_path,
        args.model_id,
        questions,
        args.answer_file,
        args.max_new_token,
        tp_size,
        embedding_type="Average"
    )
