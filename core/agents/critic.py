from langchain_core.messages import HumanMessage
from core.models import get_groq_llm


def critique_scenarios(topic: str, scenarios_text: str, research: str) -> str:
    """
    Red Team critic — qualitative only.
    Must NOT assign High/Medium/Low or numeric scores.
    """
    llm = get_groq_llm(max_tokens=900)

    prompt = f"""You are a Red Team critic reviewing forecasting scenarios.

Topic: {topic}

Research Brief:
{research}

Scenarios:
{scenarios_text}

For EACH scenario provide:
1. The weakest assumption and why it may fail
2. A material blind spot or counter-signal grounded in the research
3. One concrete improvement to the scenario or its signposts

Then propose one refined strongest scenario as plain analysis:
- what should change in the forecast
- what evidence factors should be weighted differently
- refinement rationale
- what observable development would cause you to revise the refined scenario

STRICT RULE:
Do NOT write probability labels (High/Medium/Low), percentages, or scores.
Those are computed only by the scoring pipeline, not by you.
If you want to express confidence, describe the evidence factors instead.
"""

    response = invoke_groq(
        llm,
        [HumanMessage(content=prompt)],
        max_output_tokens=900,
    )
    return response.content
