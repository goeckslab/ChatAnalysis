```mermaid
sequenceDiagram
    participant User
    participant App
    participant LLM

    User->>App: Ask a question
    App->>LLM: Send prompt with question, chat history, dataset path

    LLM-->>App: Step 1: Generate initial code
    App->>App: Run code locally

    alt Error or needs more info
        App->>LLM: Send error or code output
        LLM-->>App: Step 2: Generate next code
        App->>App: Run next step
        App->>LLM: Repeat until max_steps or final answer
    end

    App->>LLM: Send dataset head if needed

    loop Thought → Code → Observation
        LLM-->>App: Generate new code block
        App->>App: Execute code
        App->>LLM: Return observation or error
    end

    LLM-->>App: final_answer { explanation, plots, files, suggestions }
    App->>User: Display result, files, and suggestions
