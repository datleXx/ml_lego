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

from transformers import pipeline

_LOGGER = aiko.logger(__name__)
generator = pipeline(model="gpt2")

# --------------------------------------------------------------------------- #

class PE_GPT2(PipelineElement):
    def __init__(self, context):
        context.set_protocol("Llama2-7B-hf")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, text) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_GPT2: {context}, in: {text}")
        output = generator(text, do_sample=False)
        _LOGGER.info(f"PE_GPT2: {context}, out: {output}")
        return True, {"output": output}

# --------------------------------------------------------------------------- #
