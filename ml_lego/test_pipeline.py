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

_LOGGER = aiko.logger(__name__)

# --------------------------------------------------------------------------- #

class PE_Increment(PipelineElement):
    def __init__(self, context):
        context.set_protocol("increment:0")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, a) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_Increment: {context}, in a: {a}")
        a = int(a) + 1
        _LOGGER.info(f"PE_Increment: {context}, out a: {a}")
        return True, {"a": a}

# --------------------------------------------------------------------------- #

class PE_Decrement(PipelineElement):
    def __init__(self, context):
        context.set_protocol("decrement:0")
        context.get_implementation("PipelineElement").__init__(self, context)

    def process_frame(self, context, a) -> Tuple[bool, dict]:
        _LOGGER.info(f"PE_Decrement: {context}, in a: {a}")
        a = int(a) - 1
        _LOGGER.info(f"PE_Decrement: {context}, out a: {a}")
        return True, {"a": a}

# --------------------------------------------------------------------------- #
