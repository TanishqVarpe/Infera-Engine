from langchain_core.messages import HumanMessage
from core.models import get_groq_llm
from core.tools.search import web_search


def run_research_agent(topic: str, questions: list[str]) -> str:
    """
    Research Agent (stable version)
    - Manually runs web search for each question
    - Then asks Groq to synthesize the findings
    """
    llm = get_groq_llm(max_tokens=900)
    collected_info = []

    for question in questions:
        # 1. Search manually (no tool-calling issues)
        search_query = f"{question} {topic}"
        try:
            search_result = web_search.invoke(search_query)
        except Exception as e:
            search_result = f"Search failed: {str(e)}"

        collected_info.append(
            f"### Research Question\n{question}\n\n### Search Results\n"
            f"{search_result[:1800]}\n"
        )

    # 2. Synthesize with Groq
    synthesis_prompt = f"""You are an expert intelligence researcher.
Using the search results below, write a clear, factual research brief (450–550 words).

Topic: {topic}

Requirements:
- Organize around recent developments, key dates/data, relevant actors, and emerging signals.
- Distinguish reported facts from interpretation; do not add details absent from the results.
- Attribute important claims to the source title or link where possible.
- Note disagreements, weak sourcing, and important evidence gaps.
- End with 2-3 concrete indicators to watch next.
- Avoid repeating the same fact or adding generic background filler.

Search Findings:
{"".join(collected_info)}
"""

    response = invoke_groq(
        llm,
        [HumanMessage(content=synthesis_prompt)],
        max_output_tokens=900,
    )
    return response.content
