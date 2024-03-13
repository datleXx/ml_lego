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

# ---------------------LOCAL LLM IMPLEMENTATION------------------------------------------------------ #
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# import torch 

# model_id = "meta-llama/Llama-2-7b-hf"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# class PE_LLM(PipelineElement):
#     def __init__(self, context):
#         context.set_protocol("LLM")
#         context.get_implementation("PipelineElement").__init__(self, context)
#         self.system_prompt = "Your output must only be S-Expressions.  Do not provide any explanations and examples.  From now on, for all input prompts, you are a translator that turns commands into S-Expressions to control a robot dog.  The only S-Expressions that the robot dog understands are (forwards distance), (turn degrees), (wag) and (stop), (sit), (whiz).  Otherwise output (error diagnostic_message)"

#     def process_frame(self, context, text) -> Tuple[bool, dict]:
#         output = ""

#         if text is None:
#             pass
#         else:
#             text = f"{self.system_prompt}" + ". Command: " + text
#             _LOGGER.info(f"{model_id}: {context}, in: {text}")
#             model_inputs = tokenizer([f"{text}"], return_tensors="pt").to("cuda")
#             input_length = model_inputs.input_ids.shape[1]


#             # Setting `max_new_tokens` allows you to control the maximum length
#             generated_ids = model.generate(**model_inputs, max_new_tokens=50)

#             # Remove prompt inputs from the result
#             output = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

#             _LOGGER.info(f"{model_id}: {context}, out: {output}")
#         return True, {"output": output}

# ----------------------LLM THROUGH API CALL IMPLEMENTATION----------------------------------------------------- #
from gradio_client import Client

class PE_LLM(PipelineElement):
    def __init__(self, context):
        context.set_protocol("LLM")
        context.get_implementation("PipelineElement").__init__(self, context)

        self._model_id = "huggingface-projects/llama-2-7b-chat"
        self._topic_out = "aiko/speech" # static topic path to send commands to the robot
        self._client = Client(self._model_id) # API client

    def process_frame(self, context, text) -> Tuple[bool, dict]:
        output = ""
         
        # Skip inference if no audio is detected
        if text == None or text == "":
            pass
        else:
            _LOGGER.info(f"{self._model_id}: {context}, in: {text}")
            
            # LLM Hyperparameters
            max_new_tokens = 50
            temperature = 0.6
            top_p = 0.9
            top_k = 50
            repetition_penalty = 1.2

            # If the user prompt is a command
            if "command" in str.lower(text):
                system_prompt = "Your output must only be S-Expressions.  Do not provide any explanations and examples.  From now on, for all input prompts, you are a translator that turns commands into S-Expressions to control a robot dog.  The only S-Expressions that the robot dog understands are (forwards distance), (turn degrees), (wag) and (stop), (sit), (whiz).  Otherwise output (error diagnostic_message)"
                text = text.replace("command", "")

            # If the user prompt is a query
            else:
                self._client = Client(self._model_id) # Reset the API Client
                system_prompt = ""

            user_prompt = text

            # API call using Gradio Client
            output = self._client.predict(
                user_prompt,	# str  in 'Message' Textbox component
                system_prompt,	# str  in 'System prompt' Textbox component
                max_new_tokens,	# float (numeric value between 1 and 2048) in 'Max new tokens' Slider component
                temperature,	# float (numeric value between 0.1 and 4.0) in 'Temperature' Slider component
                top_p,	# float (numeric value between 0.05 and 1.0) in 'Top-p (nucleus sampling)' Slider component
                top_k,	# float (numeric value between 1 and 1000) in 'Top-k' Slider component
                repetition_penalty,	# float (numeric value between 1.0 and 2.0) in 'Repetition penalty' Slider component
                api_name="/chat"
            )

            aiko.message.publish(self._topic_out, f"(speech {len(output)}:{output}")

            _LOGGER.info(f"{self._model_id}: {context}, out: {output}")

        return True, {"output": output}
# --------------------------------------------------------------------------- #