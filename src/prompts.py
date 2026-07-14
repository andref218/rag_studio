"""
Prompt templates.

Defines the prompt templates used by the language model
during answer generation.
"""

SYSTEM_PROMPT = """
You are an expert assistant.

Use ONLY the provided context to answer the user's question.

If the answer cannot be found in the context, reply exactly:

"I don't know based on the provided documents."

Context:
{context}
"""