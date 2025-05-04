# mcp_agent_qwen.py
import os
import argparse
import asyncio
from typing import List, Dict, Optional
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Load Qwen 2.5 3B instruct model
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).eval()

# set up MCP stdio server parameters for Airbnb tools
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@openbnb/mcp-server-airbnb"],
    env=None
)

async def get_system_message(session: ClientSession) -> str:
    """ Creates a system prompt that instructs the model on how to use available Airbnb tools
    """
    tools = await session.list_tools()
    lines = []
    for t in tools.tools:
        schema = json.dumps(t.inputSchema)
        lines.append(f"- {t.name}: {t.description}. Params: {schema}")
    doc = "\n".join(lines)
    example = json.dumps({
        "name": "airbnb_search",
        "arguments": {
            "location": "Phu Quoc",
            "checkin": "2025-04-28",
            "checkout": "2025-04-30",
            "adults": 2,
            "children": 0,
            "infants": 0,
            "pets": 0,
            "minPrice": 0,
            "maxPrice": 50,
            "cursor": "",
            "ignoreRobotsText": True
        }
    }, indent=None)
    return (
        "You are an assistant with access to the following Airbnb MCP tools:\n" +
        doc + "\n" +
        "When the user asks about searching or booking, generate a JSON tool call wrapped exactly in triple backticks like this:\n" +
        "```json" + example + "```"
    )

async def generate_text(messages: List[Dict[str, str]]) -> str:
    """ Generate response from a list of messages (chat history)
    """
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = await asyncio.to_thread(
        model.generate,
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only after last 'Assistant:'
    return decoded.split("Assistant:")[-1].strip()

async def parse_and_call_tools(text: str, session: ClientSession) -> Optional[str]:
    """Search for tool call in the model's response (JSON blocks) and call the tool
    """
    # Regex to find JSON blocks
    pattern = r'```json([^`]*)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    results = []
    for block in matches:
        # Clean up whitespace/newlines
        raw = block.strip().lstrip('\n')
        try:
            call = json.loads(raw)
        except json.JSONDecodeError as e:
            results.append(f"JSON parse error: {e}")
            continue
        # Support both 'name'/'arguments' and legacy 'tool'/'params'
        tool_name = call.get("name") or call.get("tool")
        args = call.get("arguments") or call.get("params") or {}
        if not tool_name:
            results.append(f"Missing tool name in call: {call}")
            continue
        # Default robots.txt bypass if available
        if tool_name in ("airbnb_search", "airbnb_listing_details") and "ignoreRobotsText" not in args:
            args["ignoreRobotsText"] = True
        print(f"[Tool] Calling {tool_name} with {args}")
        try:
            resp = await session.call_tool(tool_name, args)
            text_out = resp.content[0].text if not resp.isError else f"Error: {resp.content[0].text}"
            results.append(f"Result from {tool_name}: {text_out}")
        except Exception as e:
            results.append(f"Tool execution error: {e}")
    return "\n".join(results)

async def run_conversation(initial_prompt: str = None):
    """ Initialize MCP client session and run a conversation
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # set up system message with available tools
            system_msg = await get_system_message(session)

            history: List[Dict[str, str]] = []
            if initial_prompt:
                history.append({"role": "user", "content": initial_prompt})

            while True:
                # 1. Read user input
                if not history or history[-1]["role"] == "assistant":
                    user = input("User: ").strip()
                    if not user or user.lower() == 'exit':
                        print("Exiting.")
                        break
                    history.append({"role": "user", "content": user})

                # 2. Generate assistant response
                msgs = [{"role": "system", "content": system_msg}] + history
                assistant = await generate_text(msgs)
                print(f"Assistant: {assistant}\n")
                history.append({"role": "assistant", "content": assistant})

                # 3. Tool execution
                tool_res = await parse_and_call_tools(assistant, session)
                if tool_res:
                    print(tool_res + "\n")
                    history.append({"role": "user", "content": tool_res})

async def main():
    parser = argparse.ArgumentParser(description="MCP agent with Qwen 2.5 and Airbnb tools.")
    parser.add_argument("-p", "--prompt", type=str, help="Initial prompt.")
    args = parser.parse_args()
    await run_conversation(args.prompt)

if __name__ == "__main__":
    asyncio.run(main())
