# Development Report: AI Prompt Generator

## 1. Overview
This software accelerates the process of creating high-quality prompts for LLM-based software development. It acts as an intelligent intermediary, clarifying user requirements through an interactive Q&A session and then structuring a comprehensive prompt based on selected development modes and "injected" expert context.

## 2. Architecture
The application is built using Python with a modular architecture:

- **`src.main`**: The CLI entry point (using `typer` and `rich`) handling user interaction.
- **`src.llm_integration`**: A wrapper around `litellm` to support multiple LLM providers (OpenAI, Anthropic, Gemini, etc.) uniformly.
- **`src.clarification_agent`**: Analyzes user intention and generates clarifying questions.
- **`src.prompt_builder`**: Assembles the final prompt, integrating:
    - User intention
    - Clarification Q&A
    - Selected Mode (One-shot, Iterative, Chain-of-Thought)
    - **Features**: Persona Injection & Complexity Estimation.

## 3. Key Features
### 3.1 Clarification Loop
The system does not just accept a raw prompt. It uses an LLM to "interview" the user, asking 3 (configurable) targeted questions to remove ambiguity before generation.

### 3.2 Persona Injection (`src/features/persona.py`)
Automatically detects the domain of the task and injects a specialized expert persona (e.g., "Senior Rust Systems Programmer") into the system prompt to guide the LLM's tone and expertise.

### 3.3 Complexity Estimation (`src/features/complexity.py`)
Analyzes the task and context to estimate difficulty (Low/Medium/High). This is included in the prompt to give the LLM context on how much detail or planning might be required.

### 3.4 Development Modes
- **One-Shot**: For simple tasks requiring a direct solution.
- **Iterative**: Encourages a plan-first, step-by-step implementation approach.
- **Chain-of-Thought**: Forces the LLM to explain its reasoning.

## 4. Verification Protocol
We employed an iterative module-by-module verification strategy:

1.  **LLM Integration**: Implemented and tested with `pytest-mock` to ensure API calls are correctly formatted without needing real keys during CI/Dev.
2.  **Clarification Agent**: Tested input/output logic and filtering of generated questions.
3.  **Prompt Builder**: Verified string assembly, persona injection, and mode-specific instructions.
4.  **CLI**: Verified entry point functionality.

**Test Coverage:**
All core logic is covered by unit tests in `tests/`.
Run tests with: `pytest`

## 5. Usage
1.  Install dependencies: `pip install -r requirements.txt`
2.  Set API keys in `.env` (e.g., `OPENAI_API_KEY=...`).
3.  Run the tool:
    ```bash
    python -m src.main
    ```
    Or with options:
    ```bash
    python -m src.main --intention "Create a snake game in Python" --model "gpt-4"
    ```

## 6. Future Work
- Integration with local LLMs via Ollama.
- Saving prompt history to a local SQLite database.
- GUI using Streamlit or Textual.
