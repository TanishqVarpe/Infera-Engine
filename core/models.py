import os
import json
from typing import TypeVar

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def get_local_llm():
    return ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.3,
    )


def get_groq_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def invoke_structured_output(
    llm: BaseChatModel,
    schema: type[StructuredModel],
    prompt: str,
) -> StructuredModel:
    structured_llm = llm.with_structured_output(schema, method="json_mode")
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    request = prompt

    for attempt in range(3):
        try:
            return structured_llm.invoke([HumanMessage(content=request)])
        except OutputParserException as error:
            if attempt == 2:
                raise ValueError(
                    f"{schema.__name__} response did not match its schema after "
                    "3 attempts."
                ) from error

            request = f"""The previous response failed schema validation.
Return a complete corrected JSON object matching this schema exactly.
Do not return Markdown, explanations, or JSON split across strings.

Schema:
{schema_json}

Validation error:
{str(error)[:3000]}

Original task:
{prompt}
"""

    raise RuntimeError("Structured response attempts ended unexpectedly.")