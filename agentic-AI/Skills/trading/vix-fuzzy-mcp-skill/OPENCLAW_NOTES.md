# OpenClaw

## Installation

Assuming you already have a ```$HOME/.openclaw/workspace/skills``` directory, copy this ```vix-fuzzy-mcp-skill``` directory into that skills directory.

This skill depends on the shared ```vix_fuzzy_shared``` package (see ```../vix_fuzzy_shared```) for VIX fetching and fuzzy set classification. Since this skill runs via bare ```python3``` (no ```uv```/venv), that package won't resolve through pip — copy its inner ```vix_fuzzy_shared/vix_fuzzy_shared/``` package directory directly alongside ```vix_fuzzy_mcp_server.py``` so Python's automatic script-directory ```sys.path``` entry picks it up:

```
cp -r ../vix_fuzzy_shared/vix_fuzzy_shared $HOME/.openclaw/workspace/skills/vix-fuzzy-mcp-skill/
```

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
  
What are the latest fuzzy VIX set memberships? Use the fuzzy VIX indicator tool to answer this question. Please also interpret these results. Thank you!

Call get_fuzzy_set_membership_of_most_recent_vix and explain the result.
