import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from threading import Thread
import time
from typing import Tuple

from aiko_services import aiko, PipelineElement
from aiko_services.utilities import generate, get_namespace, LRUCache
## ----------------------------------------------------------------------------------------------------------- ## 

class PE_LLM(PipelineElement):
    def __init__(self, context):
        context.set_protocol("llm_query:0")
        context.get_implementation("PipelineElement").__init__(self, context)
        self.device = torch.device("cuda") 

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto",  trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    def process_frame(self, context, text) -> Tuple[bool, dict]:
        if len(text) != 0: 
            inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=False)
            outputs = self.model.generate(**inputs, max_length=200)
            text_out = self.tokenizer.batch_decode(outputs)[0]
            return True, {"text": text_out}
        text_out=""
        return True, {"text": text_out}
