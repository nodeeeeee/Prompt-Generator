# Prompt Generator

This project aims to accelerate the process of generating prompts for Large Language Models (LLMs). Users can input their intention, choose a development mode, and the software will automatically generate high-quality prompts. It also includes features for clarifying requirements and connecting to various LLM APIs.

## Features (Planned)

- **User Intent Input:** Capture user's project intention.
- **Development Modes:** Support iterative, one-shot, and other custom development modes.
- **Requirement Clarification:** Ask high-quality questions to refine user's needs.
- **High-Performance Security Engine:** Advanced scanning for PII, prompt injection, and other threats with strict error handling and sanitization.
- **LLM API Integration:** Connect to Gemini, Claude, OpenAI, Deepseek, etc.
- **Prompt Template Library:** Save and reuse custom prompt templates.
- **Interactive Prompt Refinement:** Allow users to interactively improve generated prompts.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd prompt_generator
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n promptgen python=3.9
    conda activate promptgen
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Planned)

```bash
python src/main.py generate "create a simple web server with Python"
```

## Development Report

The development report will be located in the `docs/` directory.
