import re
import math
from geopy.geocoders import Nominatim
import requests
import torch
import gc
import os
from transformers import TextStreamer
from unsloth import FastLanguageModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define prompt templates for different types of questions
prompt_names = ["math", "medical", "others"]
prompt_template_math = (
    "You are an expert in solving math problems. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer. "
)

prompt_template_medical = (
    "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer. "
)

prompt_template_others = (
    f"You are great at answering all questions not from the following themes: {prompt_names[:-1]}. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer. "
)

prompt_templates = [
    prompt_template_math,
    prompt_template_medical,
    prompt_template_others,
]

# Tool functions
def bezout(a, b):
    """Compute the Bezout coefficients of a and b"""
    swap = False
    if a < b:
        swap = True
        a, b = b, a

    remainders = [a, b]
    coeff_a = [1, 0]
    coeff_b = [0, 1]
    quotients = []

    while b > 0:
        q = a // b
        a, b = b, a - b * q
        remainders.append(b)
        quotients.append(q)
        coeff_a.append(coeff_a[-2] - q * coeff_a[-1])
        coeff_b.append(coeff_b[-2] - q * coeff_b[-1])

    if swap:
        return coeff_b[-2], coeff_a[-2]
    else:
        return coeff_a[-2], coeff_b[-2]

def solve(a, b, m):
    """Solve the linear congruence equation ax=b (mod m)"""
    d = math.gcd(a, m)
    if b % d != 0:
        print("No solution!")
    else:
        a, b, m = a // d, b // d, m // d
        a_inverse = bezout(a, m)[0]
        x = a_inverse * b % m
        print(f"Solution : x = {x} (mod {m})")

def get_weather_forecast(location: str, date: str):
    """Retrieves weather forecast for a given location and date"""
    geolocator = Nominatim(user_agent="weather-app")
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}"
            )
            data = response.json()
            return {time: temp for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}

# Tool calling instruction prompt
instruction_prompt = '''
You are a specialized assistant that can perform specific computational tasks using Python functions.
Your role is to determine when to use these functions based on user requests.

Available Functions:
1. get_weather_forecast(location, date)
   - Gets hourly temperature forecast for a location on a specific date
   - Example: "What's the weather in Tokyo on 2024-05-01?"

2. bezout(a, b)
   - Computes Bezout coefficients for two integers
   - Example: "Find Bezout coefficients of 12 and 21"

3. solve(a, b, m)
   - Solves linear congruence equation ax ≡ b (mod m)
   - Example: "Solve 3x = 4 (mod 7)"

Guidelines:
- Only call functions when the request clearly matches their purpose
- Format function calls with triple backticks and "tool_code" label
- When uncertain, respond with plain text instead of making function calls
- Use the function output (labeled "tool_output") to provide helpful responses

Example:
User: "What are Bezout coefficients of 12 and 21?"
You: ```tool_code
bezout(a=12, b=21)
```
You will receive the output by user:
```tool_output
(2, -1)
```
Then you should explain: "The Bezout coefficients for 12 and 21 are 2 and -1, respectively."
'''

class Agent:
    def __init__(self, base_model_id="Qwen/Qwen2.5-3B-Instruct", adapter_path="grpo_lora", stream_output=True):
        self.messages = []
        self.stream_output = stream_output
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.model, self.tokenizer = self.load_model()
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        # Initialize default instruction prompt for tool calling
        self.messages = [{"role": "system", "content": instruction_prompt}]

    def load_model(self):
        """Load the finetuned model with proper error handling and validation"""
        # Load the base model with memory optimization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
            max_seq_length=4200,
            load_in_4bit=True,
            device_map="auto"
        )

        # Apply LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        # Load adapter if path exists
        if os.path.exists(self.adapter_path):
            model.load_adapter(self.adapter_path, adapter_name="default")
        else:
            print(f"Warning: Adapter path {self.adapter_path} not found. Using base model only.")
            
        model.eval()
        return model, tokenizer

    def classify_question_type(self, question):
        """ Classify the question type using the model
            Return: math, medical, or others
        """
        system_message = """\
        You are a text classifier. Given an input text, respond with exactly one word: math, medical, or others. Output only that word—no extra text.

        Examples:
        Text: What is the capital of Vietnam?
        Category: others

        Text: What are symptoms of Covid?
        Category: medical

        Text: How to solve a quadratic equation?
        Category: math"""

        user_message = f"Now classify this:\nText: {question}\nCategory: "
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2,
            temperature=0.0,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return response

    def extract_tool_call(self, text):
        """ Extract and execute tool code from the model's response
            Return: tool output wrapped in ```tool_output\n{output}\n```
        """
        import io
        from contextlib import redirect_stdout
        
        pattern = r"```tool_code\s*(.*?)\s*```"
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            tool_code = matches.group(1).strip()
            f = io.StringIO()
            with redirect_stdout(f):
                result = eval(tool_code)
            output = f.getvalue()
            r = result if output == '' else output
            return f'```tool_output\n{str(r).strip()}\n```'
        return None

    def should_use_tool(self, question):
        """ Determine if the question should use tool calling
            Return: True if the question should use tool calling, False otherwise
        """
        system_message = """\
        You are a text classifier. Given an input text, respond with exactly one word: tool or no_tool.
        Respond with 'tool' if the question can be answered using one of these functions:
        
        1. get_weather_forecast(location, date)
           - Gets temperature forecast for a location on a specific date
           - Example: "What's the weather in Tokyo on 2024-05-01?"
           - Use for questions about weather forecasts, temperatures, or climate conditions

        2. bezout(a, b)
           - Computes Bezout coefficients for two integers 
           - Example: "Find Bezout coefficients of 12 and 21"
           - Use for questions about finding coefficients x, y that satisfy ax + by = gcd(a,b)

        3. solve(a, b, m)
           - Solves linear congruence equation ax ≡ b (mod m)
           - Example: "Solve 3x = 4 (mod 7)"
           - Use for questions about solving modular arithmetic equations

        Otherwise respond with 'no_tool'.

        Examples:
        Text: What's the weather in Tokyo on 2024-05-01?
        Category: tool

        Text: What are Bezout coefficients of 12 and 21?
        Category: tool

        Text: Solve 3x = 4 (mod 7)
        Category: tool

        Text: What is the capital of Vietnam?
        Category: no_tool

        Text: How to solve a quadratic equation?
        Category: no_tool"""

        user_message = f"Now classify this:\nText: {question}\nCategory: "
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2,
            temperature=0.0,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return response == "tool"

    def __call__(self, message, stream=None):
        """Process a message and return the response"""
        # Use stream if provided, otherwise use instance default
        should_stream = self.stream_output if stream is None else stream
        
        # Classify the question type
        question_type = self.classify_question_type(message) # math, medical, or others
        
        # Get appropriate system prompt and temperature based on question type
        prompt_type_temperature = {"math": 0.1, "medical": 0.4, "others": 0.7}
        prompt_type_system = {
            "math": prompt_template_math,
            "medical": prompt_template_medical,
            "others": prompt_template_others
        }
        
        # Reset messages to only keep the system prompt
        self.messages = [self.messages[0]]
        
        # For medical questions, use their specific prompt
        if question_type == "medical":
            system_prompt = prompt_type_system["medical"]
            temperature = prompt_type_temperature["medical"]
            self.messages[0]["content"] = system_prompt
            formatted_message = (
                f"Question:\n{message}\n"
                "Show your reasoning in <think> </think> tags. And return the final answer in <answer> </answer> tags. "
                "Stop generating after the <answer> </answer> tags."
            )
            self.messages.append({"role": "user", "content": formatted_message})
            # execute the model
            response = self.execute(stream=should_stream, temperature=temperature)
            return response
            
        # For math and other questions, check if they need tool calling
        elif question_type in ["math", "others"]:
            if self.should_use_tool(message):
                # Use instruction prompt for tool calling
                self.messages[0]["content"] = instruction_prompt
                self.messages.append({"role": "user", "content": message})
                # 1st pass: get tool suggestion from model, don't stream this interaction
                response = self.execute(stream=None, temperature=0.7)
                # append suggestion and execute tool call from the model's response
                self.messages.append({"role": "assistant", "content": response})
                tool_code = self.extract_tool_call(response)
                if tool_code:
                    self.messages.append({"role": "user", "content": tool_code})
                    # 2nd pass: get the final response from the model, stream this interaction
                    final_response = self.execute(stream=should_stream, temperature=0.7)
                    return final_response
                return response
            else: # no tool calling
                # Use appropriate prompt for non-tool questions
                system_prompt = prompt_type_system[question_type]
                temperature = prompt_type_temperature[question_type]
                self.messages[0]["content"] = system_prompt
                formatted_message = (
                    f"Question:\n{message}\n"
                    "Show your reasoning in <think> </think> tags. And return the final answer in <answer> </answer> tags. "
                    "Stop generating after the <answer> </answer> tags."
                )
                self.messages.append({"role": "user", "content": formatted_message})
                # get model's response
                response = self.execute(stream=should_stream, temperature=temperature)
                return response

    def execute(self, stream=True, temperature=0.7):
        """Execute the model with the current conversation state"""
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True) if stream else None
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=temperature,
            streamer=streamer
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def cleanup_resources(model=None):
    """Clean up GPU memory and other resources"""
    if model is not None:
        del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    bot = None
    try:
        print("Initializing agent ...")
        bot = Agent()
        print("What can I help you with today? (Type 'exit' to quit)")
        
        while True:
            user_input = input(">> ")
            if user_input.strip().lower() == "exit":
                print("Goodbye!")
                break
                
            response = bot(user_input, stream=True)            
    finally:
        cleanup_resources(bot.model if bot else None) 