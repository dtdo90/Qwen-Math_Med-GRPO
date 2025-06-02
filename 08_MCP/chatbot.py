from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from transformers import TextStreamer
from unsloth import FastLanguageModel

import torch

import os, asyncio, json, re
import nest_asyncio
from peft import PeftModel, LoraConfig, get_peft_model

from contextlib import AsyncExitStack

nest_asyncio.apply()

# Remark: for airbnb server, need to install npm install -g @openbnb/mcp-server-airbnb

prompt_template_math = (
    "You are an expert in solving math problems. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer. "
)

prompt_template_medical = (
    "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer. "
)

prompt_template_others = (
    "You are great at answering all questions not from the following themes: math, medical. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer. "
)

class MCP_Chatbot:
    def __init__(self,base_model_id="Qwen/Qwen2.5-3B-Instruct", adapter_path="grpo_lora", stream_output=True):
        # initialize session and client objects
        self.exit_stack=AsyncExitStack()
        self.available_tools=[]
        self.available_prompts=[]
        self.stream_output = stream_output  # Initialize stream_output attribute

        # session dict maps tool_name/prompt_name/resources_URI to MCP client sessions
        self.sessions={}
        
        # initialize model
        self.base_model_id=base_model_id
        self.adapter_path=adapter_path
        self.model, self.tokenizer = self.load_model()
        

    def load_model(self):
        """ Load finetuned model"""
        # Load the base model with memory optimization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
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

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """ Connect to a single MCP server
            Extract available tools and map tool to the server session
        """
        try:
            server_params=StdioServerParameters(**server_config)
            # launch server as a subprocess with read & write permission
            stdio_transport= await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write= stdio_transport
            # initiate a session
            session=await self.exit_stack.enter_async_context(ClientSession(read,write))
            await session.initialize()

            # List available tools for this session
            tool_response=await session.list_tools()
            tools=tool_response.tools
            print(f"\nConnected to server {server_name} with tools: ", [t.name for t in tools])
            for tool in tools:
                self.sessions[tool.name]=session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
            # list available prompts
            prompt_response=await session.list_prompts()
            if prompt_response and prompt_response.prompts:
                for prompt in prompt_response.prompts:
                    self.sessions[prompt.name]=session
                    self.available_prompts.append({
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments
                    })

        except Exception as e:
            print(f"Failed to connect to server {server_name}: {str(e)}")

    async def connect_to_servers(self):
        """Connect to all MCP servers specified in server_config.json"""
        try:
            with open("server_config.json", "r") as f:
                data=json.load(f)
            servers=data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {str(e)}")
            raise
    
    async def classify_question_type(self, question):
        """ Classify the question type using the underlying model
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

        user_message = f"Now classify this text\nText: {question}\nCategory: "
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
    
    def format_schema(self, schema):
        """Format input schema to show parameters and their requirements"""
        params = []
        for name, prop in schema.get('properties', {}).items():
            required = name in schema.get('required', [])
            description = prop.get('description', 'No description available')
            param_info = f"  - {name} ({prop['type']}): {description}"
            if required:
                param_info += " [required]"
            params.append(param_info)
        return '\n'.join(params)

    async def handle_tool_usage(self, question, text=None):
        """ Handle tool detection and execution
            Args:
                question: The user's question
                text: Optional text containing tool code (response from model) to execute
            Returns:
                - If text is None: True if question should use tools, False otherwise
                - If text is provided: Tool execution result or None if no tool code found
        """
        # If text is provided, extract and execute tool code
        if text:
            pattern = r"```tool_code\s*(.*?)\s*```"
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                tool_code = matches.group(1).strip()
                try:
                    # Extract function name and arguments
                    func_name = tool_code.split('(')[0].strip()
                    args_str = tool_code[tool_code.find('(')+1:tool_code.rfind(')')]
                    # Parse arguments
                    args = {}
                    for arg in args_str.split(','):
                        if '=' in arg:
                            key, value = arg.split('=')
                            # Remove quotes from value if present
                            value = value.strip().strip('"\'')
                            args[key.strip()] = value
                    
                    # Get the session for this tool
                    if func_name in self.sessions:
                        session = self.sessions[func_name]
                        # Call the tool through MCP
                        response = await session.call_tool(func_name, args)
                        result = []
                        for part in response.content:
                            result.append(part.text)
                        
                        # Handle different types of responses
                        if len(result) == 1:
                            # Try to parse as JSON first
                            try:
                                result_json = json.loads(result[0])
                                return f'```tool_output\n{json.dumps(result_json, indent=2)}\n```'
                            except json.JSONDecodeError:
                                # If not JSON, return as is
                                return f'```tool_output\n{result[0]}\n```'
                        else:
                            # For multiple results (like bezout), return as a list
                            return f'```tool_output\n{result}\n```'
                    else:
                        return f'```tool_output\nError: Tool {func_name} not found\n```'
                except Exception as e:
                    return f'```tool_output\nError: {str(e)}\n```'
            return None
        
        # If no text provided, determine if question should use tools
        tool_descriptions = '\n'.join([
            f"{tool['name']}: {tool['description']}\nParameters:\n{self.format_schema(tool['input_schema'])}" 
            for tool in self.available_tools
        ])
        system_message = (
            "You are a text classifier. Given an input text, respond with exactly one word: tool or no_tool.\n"
            "Respond with 'tool' if the question can be answered using one of these functions:\n"
            f"{tool_descriptions}\n\n"
            "Examples:\n"
            "Question: What is the current weather in Singapore?\n"
            "Response: tool\n\n"
            "Question: What is the capital of France?\n"
            "Response: no_tool"
        )
        user_message = f"Now classify this question:\n{question}\nCategory: "
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2,
            temperature=0.0,  # deterministic answer
            do_sample=False
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return response == "tool"
    
    async def execute(self, messages, stream=True, temperature=0.7):
        """Get model's response from input messages"""
        prompt=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
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
    
    async def process_query(self, query, stream=None):
        """Process a query and return the response"""
        # Use stream if provided, otherwise use instance default
        should_stream = self.stream_output if stream is None else stream
        
        # Classify the question type
        question_type = await self.classify_question_type(query) # math, medical, or others
        question_type = question_type.lower().strip()

        # Set appropriate system prompt and temperature based on question type
        prompt_type_temperature = {"math": 0.1, "medical": 0.4, "others": 0.7}
        prompt_type_system = {
            "math": prompt_template_math,
            "medical": prompt_template_medical,
            "others": prompt_template_others
        }
        # For medical questions, use their specific prompt (no tool)
        if question_type == "medical":
            system_prompt = prompt_type_system["medical"]
            temperature = prompt_type_temperature["medical"]
            user_prompt = (
                f"Question:\n{query}\n"
                "Show your reasoning in <think> </think> tags. And return the final answer in <answer> </answer> tags. "
                "Stop generating after the <answer> </answer> tags."
            )
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            # execute the model
            response = await self.execute(messages, stream=should_stream, temperature=temperature)
            return response

        # For math and other questions, check if they need tool calling
        elif question_type in ["math", "others"]:
            if await self.handle_tool_usage(query):
                tool_descriptions = '\n'.join([
                    f"{tool['name']}: {tool['description']}\nParameters:\n{self.format_schema(tool['input_schema'])}" 
                    for tool in self.available_tools
                ])
                # set up instruction prompt
                instruction_prompt=f"""
                You are a specialized assistant that can perform specific computational tasks using the provided functions.
                Your role is to determine when to use these functions based on user requests.
                Available Functions:
                {tool_descriptions}

                Guidelines:
                - Only call functions when the request clearly matches their purpose
                - Format function calls with triple backticks and "tool_code" label
                - When uncertain, respond with plain text instead of making function calls
                - Use the function output (labeled "tool_output") to provide helpful responses
                - Check the Parameters section for each function to see required parameters and their types
                - For date parameters, use YYYY-MM-DD format
                - For numeric parameters, use numbers without quotes
                - For string parameters, use quotes

                Example:
                User: "What are Bezout coefficients of 12 and 21?"
                You: ```tool_code
                bezout(a=12, b=21)
                ```

                User: "Find me a place to stay in Singapore from June 1-5"
                You: ```tool_code
                airbnb_search(location="Singapore", check_in="2024-06-01", check_out="2024-06-05", guests=1)
                ```

                You will receive the output by user:
                ```tool_output
                user_message
                ```
                """
                # Use instruction prompt for tool calling
                messages=[
                    {"role": "system", "content": instruction_prompt},
                    {"role": "user", "content": query}
                ]

                # 1st pass: get tool suggestion from model, don't stream this interaction
                response = await self.execute(messages, stream=None, temperature=0.7)
                # append suggestion and execute tool call from the model's response
                messages.append({"role": "assistant", "content": response})
                tool_code = await self.handle_tool_usage(query, response) # get tool output
                if tool_code:
                    messages.append({"role": "user", "content": tool_code})
                    # 2nd pass: get the final response from the model, stream this interaction
                    final_response = await self.execute(messages, stream=should_stream, temperature=0.7)
                    return final_response
                return response # no tool code
            else: # no tool calling
                # Use appropriate prompt for non-tool questions
                system_prompt = prompt_type_system[question_type]
                temperature = prompt_type_temperature[question_type]
                formatted_message = (
                    f"Question:\n{query}\n"
                    "Show your reasoning in <think> </think> tags. And return the final answer in <answer> </answer> tags. "
                    "Stop generating after the <answer> </answer> tags."
                )
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_message}
                ]
                # get model's response
                response = await self.execute(messages,stream=should_stream, temperature=temperature)
                return response

    async def chat_loop(self):
        """ Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'exit' to exit.")

        while True:
            try:
                query=input(">> ").strip()
                if query.lower()=="exit":
                    break
                await self.process_query(query)
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        import gc
        # Clear model from memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()  # also clear CPU cache
        
        # Close all MCP connections
        await self.exit_stack.aclose()

async def main():
    chatbot=MCP_Chatbot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally: # clean up manually
        await chatbot.cleanup()

if __name__=="__main__":
    asyncio.run(main())
