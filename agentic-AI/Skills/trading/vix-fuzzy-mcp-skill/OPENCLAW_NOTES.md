# OpenClaw

## Installation

Assuming you already have a ```$HOME/.openclaw/workspace/skills``` directory, copy this ```vix-fuzzy-mcp-skill``` directory into that skills directory.

Run the following to register the skill with MCP:

```
openclaw mcp add vix-fuzzy-indicator \
  --command /usr/bin/python3 \
  --arg vix_fuzzy_mcp_server.py \
  --cwd /home/node/.openclaw/workspace/skills/vix-fuzzy-mcp-skill \
  --include get_fuzzy_set_membership_of_most_recent_vix
```

To test the skill registration, run:

```
openclaw mcp show vix-fuzzy-indicator
```

Then reset your session with the command ```/reset```.

## Test Prompts
  
What are the latest fuzzy VIX set memberships? Use the fuzzy VIX market indicator tool.
  
Call get_fuzzy_set_membership_of_most_recent_vix and explain the result.
