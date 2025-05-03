import re
import math
from geopy.geocoders import Nominatim
import requests

import unsloth
from transformers import TextStreamer
from unsloth import FastLanguageModel

import torch
import gc
def bezout(a,b):
    """ Compute the Bezout coefficients of a and b
        Returns:
            s,t with as+bt=gcd(a,b)
    """
    # swap a and b if necessary
    swap=False # indicate if we need to swap and b
    if a<b:
        swap=True
        a,b=b,a

    # keep track the list of remainders, coefficients a[i] and b[i], quotients q[i]
    remainders=[a,b]   # store remainders r[0], r[1], r[2], ...
    coeff_a=[1,0]      # store coefficients a[0], a[1], a[2], ...
    coeff_b=[0,1]      # store coefficients b[0], b[1], b[2], ...
    quotients=list()   # store quotients q[0], q[1], q[2], ...

    while b>0:
        # continously divide a by b and update them
        q=a//b
        a,b=b,a-b*q

        # update the lists
        remainders.append(b)
        quotients.append(q)
        coeff_a.append(coeff_a[-2]-q*coeff_a[-1])
        coeff_b.append(coeff_b[-2]-q*coeff_b[-1])

    if swap:
        return coeff_b[-2], coeff_a[-2]
    else:
        return coeff_a[-2], coeff_b[-2]

def solve(a,b,m):
    """ Solve the linear congruence equation ax=b (mod m)
        Returns: 
            print out solution in the form x=c (mod n)
    """
    d=math.gcd(a,m)

    # no solution if b is not divisible by d
    if b%d!=0:
        print("No solution!")

    # divide a,b,m by d and solve the resulting equation
    else:
        a,b,m=a//d, b//d, m//d
        a_inverse=bezout(a,m)[0]
        x=a_inverse*b % m
        print(f"Solution : x = {x} (mod {m})")


def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a list dictionary with the time and temperature for each hour."""
    geolocator = Nominatim(user_agent="weather-app") 
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}")
            data = response.json()
            return {time: temp for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}


# few-shot instruction prompt
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
   - Example: "Solve 3x ≡ 4 (mod 7)"

Guidelines:
- Only call functions when the request clearly matches their purpose
- Format function calls with triple backticks and "tool_code" label
- When uncertain, respond with plain text instead of making function calls
- Use the function output (labeled "tool_output") to provide helpful responses

Example for Bezout:
User: "What are Bezout coefficients of 12 and 21?"
You: ```tool_code
bezout(a=12, b=21)
```
You will receive: ```tool_output
(2, -1)
```
Then you should explain: "The Bezout coefficients for 12 and 21 are 2 and -1, respectively."

Example for weather:
User: "What's the weather in Tokyo on 2024-05-01?"
You: ```tool_code
get_weather_forecast(location="Tokyo", date="2024-05-01")
```
You will receive: ```tool_output
{'2024-05-01T00:00': 15.2, '2024-05-01T01:00': 14.8, ...}
```
Then you should explain: "The temperature in Tokyo on 2024-05-01 ranges from 14.8°C to 15.2°C."
'''

# Another prompt type
# instruction_prompt= '''
# You are a specialized assistant provided with a set of Python functions to perform specific computational tasks. Your task is to decide whether to answer the user's request with a plain text response or to call one of the provided functions.
# At each turn, if you decide to invoke any of the functions, you must wrap the function call with triple backticks and the label "tool_code". The generated code should be readable, efficient, and match exactly the provided function signature.
# When a function call is made, its output will be wrapped in triple backticks with the label "tool_output". Use that output to guide any further tool calls or to generate a friendly response.

# Guidelines:
# 1. Only call a function if the user's request clearly maps to one of the functions below.
# 2. In cases of ambiguity, always choose to answer in plain text rather than risk an unnecessary function call.
# 3. When you decide to call a function, format your call with triple backticks labeled "tool_code".

# For example, if the user asks: "What are bezout coefficients of 12 and 21?" then you must respond with:
# ```tool_code
# bezout(a=12, b=21)
# ```


# The available Python methods are:

# ```python
# def get_weather_forecast(location: str, date: str):
#     """ Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). 
#         Returns a list dictionary with the time and temperature for each hour.

#         Args:
#           location: the location for which to get the weather
#           data: the forecasting date for when to get the weather format (yyyy-mm-dd)
#     """

# def bezout(a:int , b: int) -> List[int]:
#     """Compute the Bezout coefficients of a and b

#     Args:
#       a: The first number
#       b: The second number
#     """

# def solve(a,b,m):
#     """Solve the linear congruence equation ax=b (mod m)

#     Args:
#       a: The coefficient of x in the equation
#       b: The right-hand side of the equation
#       m: The modulus of the equation
#     """
# ```
# '''



def extract_tool_call(text):
  """Extract output of calling the tool code from the response
    Args:
      text: the response from the model
    Returns:
      The output of the tool code, for example:
      ```tool_output
        (2, -1)
      ```
  """
  import io
  from contextlib import redirect_stdout
  # pattern: tool_code + white spaces + character + white spaces
  pattern=r"```tool_code\s*(.*?)\s*```"
  matches=re.search(pattern, text, re.DOTALL)
  if matches:
      tool_code=matches.group(1).strip()
      # capture stdout in a string buffer: to capture the output generated by the function eval(tool_code), e.g. function contains print statement
      f=io.StringIO()
      # eval executes the extracted code as if it were a Python expression
      with redirect_stdout(f):
          result=eval(tool_code)
      output=f.getvalue()
      r=result if output== '' else output
      return f'```tool_output\n{str(r).strip()}\n```'
  return None


class Agent:
    def __init__(self, base_model_id="Qwen/Qwen2.5-3B-Instruct", system_message=instruction_prompt, stream_output=True):
        self.messages = []
        self.stream_output = stream_output
        self.base_model_id=base_model_id
        self.model, self.tokenizer=self.load_model()
        if system_message:
            self.messages = [{"role": "system", "content": system_message}]

    def load_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
            max_seq_length=2048,         
            load_in_4bit=True,           
            )
        return model, tokenizer
    
    def __call__(self, message, stream=None):
        # Use parameter stream if provided, otherwise use instance default
        should_stream = self.stream_output if stream is None else stream
        
        # user message
        self.messages.append({"role": "user", "content": message})
        response = self.execute(stream=should_stream)

        # Add the assistant's response to the message history
        self.messages.append({"role": "assistant", "content": response})
        return response

    def execute(self, stream=True):
        # apply chat template 
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        # Only use TextStreamer if streaming is enabled
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True) if stream else None
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            # temperature=0.1,  # Control randomness (lower = more deterministic)
            # top_p=0.9,   
            streamer=streamer
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    

def query(bot, question):
    # get tool call - stream this first interaction
    response = bot(question, stream=True)     
    # get output of tool function
    call_response = extract_tool_call(response)
    if call_response is None:
        return None
    print(call_response)

    # don't stream the second interaction
    answer = bot(call_response, stream=False)
    return answer


# cleanup resources
def cleanup_resources(model=None):
    """Clean up GPU memory and other resources"""
    if model is not None:
        del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__=="__main__":
    try:
        print("Initializing agent ...")
        bot = Agent()
        print("What can I help you with today? (Type 'exit' to quit)")
        while True: 
            # read user input
            user_input = input(">> ")
            if user_input.strip().lower() == "exit":
                print("Goodbye!")
                break
            answer = query(bot, user_input)
            print(answer)
    finally:
        cleanup_resources(bot.model)
