import json
import requests

import pandas as pd
from vllm import LLM, SamplingParams
from pyauth import Authentication

api_key = ""

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

def post_process_entities(entities):
    """
    Post-process extracted entities from the model output.

    Example list of entities:
    [
        {'phrase': 'allergies', 'semantic_type': 'Disease, Syndrome or Pathologic Function', 'assertion_status': 'title', 'body_location': 'null', 'modifier': 'null', 'value': 'not applicable', 'unit': 'not applicable', 'purpose': 'not applicable'}, 
        {'phrase': 'sulfur', 'semantic_type': 'Chemical or Drug', 'assertion_status': 'present', 'body_location': 'not applicable', 'modifier': 'not applicable', 'value': 'null', 'unit': 'units: null', 'purpose': 'null'}, 
        {'phrase': 'norvasc', 'semantic_type': 'Chemical or Drug', 'assertion_status': 'present', 'body_location': 'not applicable', 'modifier': 'not applicable', 'value': 'null', 'unit': 'units: null', 'purpose': 'null'}, 
        {'phrase': 'abdominal pain', 'semantic_type': 'Sign, Symptom, or Finding', 'assertion_status': 'present', 'body_location': 'Abdominal', 'modifier': 'null', 'value': 'not applicable', 'unit': 'not applicable', 'purpose': 'not applicable'}, 
        {'phrase': 'surgical or invasive procedure', 'semantic_type': 'Therapeutic or Preventive Procedure', 'assertion_status': 'title', 'body_location': 'null', 'modifier': 'not applicable', 'value': 'not applicable', 'unit': 'not applicable', 'purpose': 'null'}
    ]

    Args:
        entities (list): List of entity dictionaries extracted from model responses

    Returns:
        list: List of processed entity dictionaries with standardized formatting
    """
    processed_entities = []
    
    for entity in entities:
        # Get CUI from UMLS API for the phrase
        cui = get_cui_from_umls(entity['phrase'])
        
        # Add CUI to entity dictionary if found
        entity['cui'] = cui if cui else 'null'
            
        processed_entities.append(entity)
    
    return processed_entities


def get_cui_from_umls(term):
    """
    Query UMLS API to get CUI for a given term.
    
    Args:
        term (str): Medical term to search for in UMLS
        
    Returns:
        str: CUI code if found, None otherwise
    """
    # Get authentication token using pyauth helper
    auth_client = Authentication(api_key)
    tgt = auth_client.gettgt()
    service_ticket = auth_client.getst(tgt)
    
    # UMLS API endpoint
    base_url = 'https://uts-ws.nlm.nih.gov/rest'
    version = 'current'
    search_url = f'{base_url}/search/{version}'
    
    # Parameters for the search
    params = {
        'string': term,
        'ticket': service_ticket,
        'searchType': 'exact',
        'returnIdType': 'CUI',
        'pageSize': 1  # We only need the first/best match
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        results = response.json()
        
        # Check if we got any results
        if (results.get('result') and 
            results['result'].get('results') and 
            len(results['result']['results']) > 0):
            
            return results['result']['results'][0]['ui']
            
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error querying UMLS API for term '{term}': {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing UMLS API response for term '{term}': {e}")
        return None

if __name__ == "__main__":
    responses = process_ehr_data()
    print(responses)
    # res = json.loads(output[0])


