# MindGames Agent

A language model agent system designed for strategic game environments with automatic prompt routing and game-specific optimizations.

## Requirements

- Python 3.10 – 3.12
- OpenAI API key

All dependencies are pinned in `requirements.txt` with specific versions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yinita/mindgames_agent.git
   cd mindgames_agent
   ```

2. Create virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Python API

```python
from src.agent import agent

# Create agent with automatic prompt routing
bot = agent("openai", model_name="gpt-4o-mini")

# Use the agent
observation = "[GAME] Night phase - choose one player to investigate: [0], [2], [3]"
action = bot(observation)
print(action)
```

### Command Line Interface

Single entry point for running matches and tests:

```bash
python -m src.main --model-name gpt-4o --observation "[GAME] Welcome to Secret Mafia!..."
```

**CLI Arguments:**
- `--model-name` (required): OpenAI model to use (e.g., gpt-4o, gpt-4o-mini)
- `--observation` (required): Game observation string
- `--backend` (optional): Backend type, defaults to "openai"
- `--system-prompt` (optional): Custom system prompt override

**Example:**
```bash
python -m src.main --model-name gpt-4o-mini --observation "[GAME] You are the Sheriff. Choose someone to investigate: [1], [2], [3]"
```

## Testing

Run the test suite:
```bash
pytest
```

## Project Structure

```
├── src/
│   ├── agent.py              # Main API entry point
│   ├── agents/
│   │   └── agent.py          # Agent base class and OpenAI implementation
│   ├── utils/
│   │   └── prompts.py        # Prompt loading utilities
│   └── main.py               # CLI entry point
├── configs/
│   └── prompts/              # Game-specific prompts
│       ├── secret_mafia/
│       ├── colonel_blotto/
│       └── ...
├── tests/                    # Test suite
├── requirements.txt          # Pinned dependencies
└── README.md                # This file
```

## Features

- **Automatic Prompt Routing**: Detects game type from observations and selects appropriate prompts
- **Game-Specific Optimization**: Tailored strategies for different game environments
- **Modular Design**: Easy to extend with new games and agent types
- **CLI Interface**: Simple command-line access for evaluation and testing
