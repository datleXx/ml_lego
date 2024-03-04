# Usage
# ~~~~~
# aiko_pipeline create pipeline_test.json
#
# export AIKO_LOG_MQTT=false
# TOPIC=$NAMESPACE/$HOST/$PID/$SID/in
# mosquitto_pub -h $HOST -t $TOPIC -m "(process_frame (stream_id: 0) (a: 0))"
#
# To Do
# ~~~~~
# - None, yet !

from typing import Tuple

from aiko_services import aiko, PipelineElement

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch 

_LOGGER = aiko.logger(__name__)

model_id = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# --------------------------------------------------------------------------- #

class PE_GPT2(PipelineElement):
    def __init__(self, context):
        context.set_protocol("Llama2-7B-hf")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, text) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_GPT2: {context}, in: {text}")
        model_inputs = tokenizer([f"{text}"], return_tensors="pt").to("cuda")

        generated_ids = model.generate(**model_inputs)
        tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Setting `max_new_tokens` allows you to control the maximum length
        generated_ids = model.generate(**model_inputs, max_new_tokens=50)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        _LOGGER.info(f"PE_GPT2: {context}, out: {output}")
        return True, {"output": output}

# --------------------------------------------------------------------------- #
