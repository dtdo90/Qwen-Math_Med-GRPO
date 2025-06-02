# Math and Medical Chatbot with MCP Servers

This project implements a sophisticated chatbot system that can handle mathematical calculations, medical queries, and general questions using Model Context Protocol (MCP) servers. The system uses a fine-tuned Qwen2.5-3B-Instruct model with LoRA adapters for improved performance.

## Quick Start

1. Install UV and setup environment:
```bash
pip install uv
uv venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows

uv pip install -r requirements.txt
npm install -g @openbnb/mcp-server-airbnb
```

2. Run the chatbot:
```bash
uv run chatbot.py
```

## Components

- **Chatbot**: Main interface (`chatbot.py`)
- **MCP Servers**:
  - Math Server: Bezout coefficients, linear congruence equations
  - Weather Server: Weather information
  - Airbnb Server: Airbnb data (requires npm)

## Features

- Automatic question classification (math/medical/others)
- Multiple MCP server integration
- Streaming output
- LoRA fine-tuned model
- Async server communication

## Configuration

Modify `server_config.json` to:
- Add/remove servers
- Change server parameters
- Update command arguments
- Set environment variables
