from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_answer(query: str, contexts: list[str], model_name: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    joined_context = "\n\n---\n\n".join(contexts)

    prompt = f"""
              You are a technical assistant answering questions about a software repository.
  
              Answer using only the retrieved context below.
              If the context is insufficient, say so clearly.
              Cite file-level evidence when possible.
  
              Question:
              {query}
  
              Retrieved Context:
              {joined_context}
              """

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""
