import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from threading import Thread
import time
from typing import Tuple

from aiko_services import aiko, PipelineElement
from aiko_services.utilities import generate, get_namespace, LRUCache

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

## ----------------------------------------------------------------------------------------------------------- ## 

class PE_LLM(PipelineElement):
    def __init__(self, context):
        context.set_protocol("llm_query:0")
        context.get_implementation("PipelineElement").__init__(self, context)

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    def process_frame(self, context, text) -> Tuple[bool, dict]:
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_length=200)
        text_out = self.tokenizer.batch_decode(outputs)[0]
        return True, {"text": text_out}
