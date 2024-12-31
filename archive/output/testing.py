import argparse
import json
import os
import time

import numpy as numpy
import pandas as pd
import torch
import transformers
from fastchat.conversation import (
    Conversation,
    SeparatorStyle,
    get_conv_template,
    register_conv_template,
)
from fastchat.model import get_conversation_template
from fastchat.model.model_adapter import BaseModelAdapter, register_model_adapter
from tqdm import tqdm
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    template_all = [
        "Please provide your response based on the information in the electronic health record:\n {question} \nThe information in the electronic health record is as follows:\n {note}",
        "To answer the following question, please refer to the electronic health record: \n{note}\n\nThe question is as follows:\n{question}",
        "Please respond to the following question based on the electronic health record entry provided:\n\nElectronic Health Record:\n{note}\n\nQuestion:\n{question}",
        "Based on the information stored in the electronic health record, could you kindly provide a response to the following inquiry:\n{question}\nThe relevant notes are as follows:\n{note}",
        "To address the following question, please refer to the electronic health record:\n{note}\n\nQuestion:\n {question}",
        "Please provide a response to the following question based on the information presented in the electronic health record:\n\n{question}\n\nThe relevant information is provided in the following note: \n\n{note}\n",
        "Given the electronic health record:\n {note} \n\nPlease answer this question accordingly:\n {question}",
    ]
    model_path = (
        "/n/data1/hsph/biostat/celehs/lab/hongyi/ehrllm/0212-llama27b-ehrqa-1epoch"
    )
    # Load the tokenizer and model

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto"
    )

    note = """
    SICU Nsg ADmit Note
     Pt arrived to the sicu from [**Location (un) 982**] [**Hospital 983**] Hosp via EW at approx 6pm.  She is an 80 y/o female diagnosed w/ non-small cell ca of the right paratracheal region and the left base Noted on X-ray on [**3-6**].  She had presented w/ sx's of SOB, weight loss and cough.  A repeat CT showed progression of the mass w/ compression on the distal trachea and right mainstem bronchus on [**3-22**], she was immediately tx'd w/ radiation and chemo at NWH x's 5 days yet con't w/ worsening symptoms and was transferred to [**Hospital1 984**] for a rigid bronch w/ stent placements.
     Pt went to the OR last eve for stent placements in the distal trachea and the right mainstem bronchus.   Also getting decadron Q4hrs for swelling and bronchospasm.

  Other PMx/PSHx:  hx smoking, HTN, PVDz w/ Left BKA in "83, and fem [**Doctor Last Name 419**] in '[**26**] w/ redo in '[**40**], H/O PE s/p IVC filter, s/p ventral hernia repair.
  Alleries:  sulfa-hives
  Pre-op meds:  dilt, albuteral/atrovent, decadron, levaquin, coumadin, maxide, zofran.

current ROS

[**Name (NI) **]  pt alert oriented x's [**1-28**], appropriate and co-operative,

[**Name (NI) 78**] pt in SR, tachycardic at times w/ occaisional ectopy, bigem, couplets.  Lytes wnl's.  Hct 32. distal pulses palp.

[**Name (NI) **] pt on 3-4l nc and 50% FT pre and post procedure, sats from 91-98%, strong congested cough, very rhoncerous w/ some exp wheezes in upper lobes.  Very air hungry and restless w/ desaturation upon  arrival from stent placement, pt tachypnic and increasingly tachycardic at that time as well, attempted repositioning yet also gave her a neb tx and some mso4 w/ good results, pt settled and was able to get rest.

gi- Abd soft, bowel sounds hypoactive, taking small amts clear liqs w/ meds.

GU- foley placed in SICU on admission, u/o marginal over night at 20-45 cc's hr, currently getting IVF of .9%NS at 100cc's hr.  Pt on PO protonix.

Heme- hct stable, pt started on SQ hep for profolaxis at this time.

endo- bs slightly elevated on decadcom.

Code status discussed w/ family last night, decided that pt would be a DNR/DNI.

Social- one son and a daughter in to see pt.

A/P- con't to offer supportive care as ordered, pulmonary toilet as tolerated.

Endo- blood sugar 150 on decadron,
    """
    question = (
        "Is this patient a current smoker, past smoker, or has never smoked before?"
    )
    input_texts = f"Given the electronic health record:\n {note} \n\nPlease answer this question accordingly:\n {question}"

    device = torch.device("cuda")
    model.to(device)

    input_ids = tokenizer(input_texts, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True, output_hidden_states=True)

    print(outputs)
    predicted_token_ids = outputs.logits.argmax(-1)
    output_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

    print(output_text)
    # The start and end positions of the answer in the context
    # answer_start = torch.argmax(outputs.start_logits)
    # answer_end = torch.argmax(outputs.end_logits) + 1

    # # Convert tokens to answer string
    # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    # print("Answer:", answer)
