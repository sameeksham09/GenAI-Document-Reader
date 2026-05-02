RAG-based Question Answering System
An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload documents, ask different types of questions, and receive grounded, document-based answers with citations through an interactive UI.

🚀 Features Implemented
 1. Retrieval-Augmented Generation (RAG)
    - Uses FAISS vector search to retrieve the most relevant document chunks.
    - Employs Sentence Transformers (all-MiniLM-L6-v2) for semantic embeddings.
    - Ensures answers are generated only from retrieved context (no hallucinations).
    - Includes similarity thresholding to avoid low-quality answers.

 2. Multiple Question Modes
    Users can choose different types of questions:
    Descriptive - > Generates a direct, grounded answer
    MCQ - > Generates multiple-choice questions with correct answers
    True / False - > Generates multiple True/False questions with justification
    Fill in the Blanks - > Creates fill-in-the-blank questions with answers

   Users can dynamically select how many MCQs or True/False questions to generate.    

3. Dynamic Document Upload (TXT & PDF)
    - Users can upload TXT or PDF files directly from the UI.
    - Documents are :
        * Parsed(PDF-Text)
        * Chunked Intelligently
        * Embedded and stored in FAISS
    - Knowledge base updates in real-time without restarting the app.

4. Interactive Streamlit UI
    - Clean and user-friendly Streamlit interface
    - Dynamic UI elements:
        * Question placeholders change based on selected mode
        * Question count selector appears only for relevant modes
    - Displays :
        * AI-generated answers
        * Source document names
        * Similarity scores for transparency

5. Prompt Engineering & Guardrails
    - Strict prompt constraints:
        * Uses only retrieved context
        * Prevents prior knowledge usage
        * Responds with "I don't know based on the provided document" when context is insufficient
    - Separate prompt logic for each question type.

6. LLM backends (set `LLM_BACKEND` before running)
    - **ollama** (default): local model via Ollama; no PyTorch in the app process.
    - **openai**: OpenAI API; set `OPENAI_API_KEY`; may hit rate limits.
    - **local_lora**: your fine-tuned TinyLlama+LoRA from `Training/lora-model`.
      Use after training with `create_training_dataset.py` → `convert_to_instruction.py` → `Training/train_lora.py`.
      On some macOS setups PyTorch can crash; use on Linux/Colab or stick to `ollama`/`openai`.

7. Evaluation
    - Held-out test set: `evaluation/test_qa.json` (add your own `question` + `reference` pairs).
    - Run from project root: `python evaluation/run_eval.py`
    - Reports **token F1** and **exact match** vs references. Optional: `EVAL_OUTPUT=results.json` to save results.