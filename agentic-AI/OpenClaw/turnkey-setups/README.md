# Turnkey OpenClaw Configurations

This set of tools allows one to spin up OpenClaw using a previously defined configuration. No messing!

## Configure Environment Variables

Set "BRAVE_API_KEY" to your Brave API key:

```
export BRAVE_API_KEY=...
```

Set "OPENCLAW_HOME" to the directory OpenClaw code directory that you will pull from GitHub:

```
export OPENCLAW_HOME=/../../../openclaw
```

In this directory, you will find one or more turnkey OpenClaw setup. Select the one you want by setting the "OPENCLAW_TURNKEY_SETUP_HOME" environment variable, e.g., 

```
export OPENCLAW_TURNKEY_SETUP_HOME=<path to this directory>/docker-local-ollama
```

Set "OPENCLAW_DEFAULT_LLM" to the default (installed) Ollama LLM you want to use, e.g.:

```
export OPENCLAW_DEFAULT_LLM=qwen3.6:35b
```

## Shut Down a Running Pre-Existing Configuration

The following only works for OpenClaw configurations initialized using one of these turnkey setups:

```
cd $OPENCLAW_HOME

docker compose down

export DOCKER_IMAGE_ID=$(docker images -q "openclaw:local")

docker rmi $DOCKER_IMAGE_ID
```

## Start OpenClaw Using a Turnkey Configuration

```
cd $OPENCLAW_HOME/..

rm -R -f $OPENCLAW_HOME

git clone https://github.com/openclaw/openclaw.git

cd $OPENCLAW_HOME

rm .env

cat $OPENCLAW_TURNKEY_SETUP_HOME/append-to-Dockerfile >> Dockerfile

docker build -t openclaw:local -f Dockerfile .

papermill $OPENCLAW_TURNKEY_SETUP_HOME/configure-OpenClaw.ipynb /tmp/configure-OpenClaw.OUTPUT.ipynb

docker compose up -d openclaw-gateway

export CONTAINER_ID=$(docker ps -aqf "ancestor=openclaw:local")

docker exec -it $CONTAINER_ID bash

openclaw doctor --fix

openclaw completion --shell zsh --install --yes

openclaw gateway restart

exit
```

## Access the OpenClaw Webapp

[http://127.0.0.1:18789/](http://127.0.0.1:18789/)
