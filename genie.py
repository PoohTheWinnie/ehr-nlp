import json
import time

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

    Example:
        >>> ehr_responses = process_ehr_data(
        ...     model_path="path/to/model",
        ...     data_path="data/ehr_records.csv",
        ...     n_samples=5
        ... )
        >>> print(ehr_responses[0])
        [{'phrase': 'abdominal pain', 'semantic_type': 'Sign, Symptom, or Finding', ...}]
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
        last_closed_brace_index = response.rfind("}")
        complete_json_string = response[: last_closed_brace_index + 1] + "]"
        responses.append(json.loads(complete_json_string))

    return responses


def post_process_entities(entities, cui_dictionary_file, cui_column="cui", term_column="term"):
    """
    Post-process extracted entities from the model output using a CSV dictionary file.

    Args:
        entities (list): List of entity dictionaries extracted from model responses
        cui_dictionary_file (str): Path to CSV file containing CUI mappings
        cui_column (str): Name of the column containing CUI codes in the CSV
        term_column (str): Name of the column containing terms in the CSV

    Returns:
        list: List of processed entity dictionaries with standardized formatting

    Example:
        >>> entities = [{
        ...     'phrase': 'abdominal pain',
        ...     'semantic_type': 'Sign, Symptom, or Finding',
        ...     'assertion_status': 'present'
        ... }]
        >>> processed = post_process_entities(
        ...     entities,
        ...     cui_dictionary_file='data/cui_dictionary.csv'
        ... )
        >>> print(processed[0])
        {'phrase': 'abdominal pain', 'semantic_type': 'Sign, Symptom, or Finding', 
         'assertion_status': 'present', 'cui': 'C0000737'}
    """
    start_time = time.time()

    # Read only necessary columns from CSV
    cui_df = pd.read_csv(cui_dictionary_file, usecols=[cui_column, term_column])

    # Create term_to_cui mapping once
    term_to_cui = dict(zip(cui_df[term_column].str.lower(), cui_df[cui_column]))

    # Pre-extract all phrases for faster lookup
    entity_phrases = {entity["phrase"].lower() for entity in entities}

    # Create a filtered mapping for only relevant terms
    relevant_terms = term_to_cui.keys() & entity_phrases
    filtered_term_to_cui = {term: term_to_cui[term] for term in relevant_terms}

    # Process entities using list comprehension
    processed_entities = [
        {**entity, "cui": filtered_term_to_cui[entity["phrase"].lower()]}
        for entity in entities
        if entity["phrase"].lower() in filtered_term_to_cui
    ]

    end_time = time.time()
    print(
        f"Post-processing took {end_time - start_time:.2f} seconds for {len(entities)} entities"
    )
    print(f"Matched {len(processed_entities)} entities with CUIs")

    return processed_entities





if __name__ == "__main__":
    # Example usage of process_ehr_data() and get_cui_from_umls()

    # Example 1: Get CUI code for a medical term
    # example_term = "abdominal pain"
    # cui_code = get_cui_from_umls(example_term)
    # print(f"\nExample 1: Getting CUI code")
    # print(f"Term: {example_term}")
    # print(f"CUI Code: {cui_code}")

    # Example 2: Process some sample EHR data
    # sample_ehr_text = """
    # Patient presents with fever and cough for 3 days.
    # Medical History: Type 2 Diabetes, Hypertension
    # Medications: Metformin 500mg BID, Lisinopril 10mg daily
    # """
    # print("\nExample 2: Processing EHR text")
    # print("Input text:")
    # print(sample_ehr_text)
    # print("\nProcessed entities:")
    # responses = process_ehr_data()
    # print(responses)

    # Example 3: Post processing function

    entities = [
        {
            "phrase": "allergies",
            "semantic_type": "Disease, Syndrome or Pathologic Function",
            "assertion_status": "title",
            "body_location": "null",
            "modifier": "null",
            "value": "not applicable",
            "unit": "not applicable",
            "purpose": "not applicable",
        },
        {
            "phrase": "sulfur",
            "semantic_type": "Chemical or Drug",
            "assertion_status": "present",
            "body_location": "not applicable",
            "modifier": "not applicable",
            "value": "null",
            "unit": "units: null",
            "purpose": "null",
        },
        {
            "phrase": "norvasc",
            "semantic_type": "Chemical or Drug",
            "assertion_status": "present",
            "body_location": "not applicable",
            "modifier": "not applicable",
            "value": "null",
            "unit": "units: null",
            "purpose": "null",
        },
        {
            "phrase": "abdominal pain",
            "semantic_type": "Sign, Symptom, or Finding",
            "assertion_status": "present",
            "body_location": "Abdominal",
            "modifier": "null",
            "value": "not applicable",
            "unit": "not applicable",
            "purpose": "not applicable",
        },
        {
            "phrase": "surgical or invasive procedure",
            "semantic_type": "Therapeutic or Preventive Procedure",
            "assertion_status": "title",
            "body_location": "null",
            "modifier": "not applicable",
            "value": "not applicable",
            "unit": "not applicable",
            "purpose": "null",
        },
        {
            "phrase": "renovascular hypertension",
            "semantic_type": "Disease, Syndrome or Pathologic Function",
            "assertion_status": "present",
            "body_location": "renal",
            "modifier": "null",
            "value": "not applicable",
            "unit": "not applicable",
            "purpose": "not applicable",
        },
        {
            "phrase": "non-st elevation myocardial infarction",
            "semantic_type": "Disease, Syndrome or Pathologic Function",
            "assertion_status": "present",
            "body_location": "null",
            "modifier": "null",
            "value": "not applicable",
            "unit": "not applicable",
            "purpose": "not applicable",
        },
    ]

    post_process_entities(
        entities * 143000, cui_dictionary_file="data/cui_dictionary.csv"
    )

    # res = json.loads(output[0])
