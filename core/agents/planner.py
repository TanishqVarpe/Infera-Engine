from langchain_core.messages import HumanMessage
from core.models import get_groq_llm


def plan_research(topic: str) -> list[str]:
    """Planner Agent: Decides the key research questions."""
    llm = get_groq_llm(max_tokens=300)

    prompt = f"""You are a senior intelligence planner.
Break down the following developing topic into 3 distinct, investigative research questions for high-quality forecasts:
1. What has changed recently, and what is the best evidence?
2. Which actors, incentives, or constraints could drive the next developments?
3. What signals or conditions would distinguish the most plausible outcomes?

Topic: {topic}

Return exactly 3 specific questions, one per line. Avoid repeating the same angle.
"""

    response = invoke_groq(
        llm,
        [HumanMessage(content=prompt)],
        max_output_tokens=300,
    )
    questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return questions[:3]
