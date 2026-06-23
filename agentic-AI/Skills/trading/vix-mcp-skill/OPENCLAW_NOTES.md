# OpenClaw

## Installation

Assuming you already have a ```$HOME/.openclaw/workspace/skills``` directory, copy this ```vix-mcp-skill``` directory into that skills directory.

Run the following to register the skill with MCP:

```
openclaw mcp add vix-market-indicator \
  --command /usr/bin/python3 \
  --arg vix_mcp_server.py \
  --cwd /home/node/.openclaw/workspace/skills/vix-mcp-skill \
  --include get_most_recent_vix
```

To test the skill registration, run:

```
openclaw mcp show vix-market-indicator
```

Then reset your session with the command ```/reset```.

## Test Prompts
  
What is the latest VIX value? Use the VIX market indicator tool.
  
Call get_most_recent_vix with period 7d and explain the color result.

