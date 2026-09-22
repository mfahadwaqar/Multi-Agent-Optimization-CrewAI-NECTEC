"""Optional forecasting CrewAI layer; prompts never contain raw observations."""

import json
from typing import Any

from agent_team import _llm


def _crewai():
    try:
        from crewai import Agent, Crew, Process, Task
        return Agent, Crew, Process, Task
    except ImportError as exc:
        raise RuntimeError("CrewAI is not installed. Run: pip install -r requirements.txt") from exc


def review_forecast(profile: dict[str, Any], results: list[dict[str, Any]], best: dict[str, Any]) -> str:
    Agent, Crew, Process, Task = _crewai()
    llm = _llm()
    seasonality_analyst = Agent(
        role="Time-Series and Seasonality Analyst",
        goal="Check frequency, horizon, gaps and plausible seasonal structure from the derived profile.",
        backstory="You never request or receive raw observations.", llm=llm, verbose=False,
    )
    leakage_auditor = Agent(
        role="Forecast Leakage Auditor",
        goal="Check that features, gaps and evaluation use only information available at prediction time.",
        backstory="You distinguish chronological forecasting from random classification splitting.", llm=llm, verbose=False,
    )
    model_analyst = Agent(
        role="Forecast Model Analyst",
        goal="Compare measured walk-forward MAE, RMSE and MASE without inventing results.",
        backstory="Lower validation error determines selection; test results are final evidence only.", llm=llm, verbose=False,
    )
    critic = Agent(
        role="Weather and Water Forecast Critic",
        goal="Identify limitations and practical next experiments from the supplied summaries.",
        backstory="You make no causal or operational safety claims from predictive metrics alone.", llm=llm, verbose=False,
    )
    reporter = Agent(
        role="Forecast Experiment Reporter",
        goal="Explain the measured result briefly in plain language.",
        backstory="You preserve all stated limitations and metric meanings.", llm=llm, verbose=False,
    )
    payload = json.dumps({"profile": profile, "validation_results": results, "selected_result": best}, indent=2)
    season_task = Task(description="Assess this derived forecasting profile:\n" + payload, expected_output="A short seasonality and data-quality assessment.", agent=seasonality_analyst)
    leakage_task = Task(description="Audit the described pipeline for temporal leakage.", expected_output="A short leakage assessment.", agent=leakage_auditor, context=[season_task])
    model_task = Task(description="Interpret only the measured model metrics in the payload.", expected_output="A short model comparison.", agent=model_analyst, context=[season_task])
    critic_task = Task(description="State the most important limitations and next check.", expected_output="A concise critique.", agent=critic, context=[leakage_task, model_task])
    report_task = Task(description="Write at most 120 words in simple language.", expected_output="A concise final review.", agent=reporter, context=[leakage_task, model_task, critic_task])
    return str(Crew(agents=[seasonality_analyst, leakage_auditor, model_analyst, critic, reporter], tasks=[season_task, leakage_task, model_task, critic_task, report_task], process=Process.sequential, verbose=False).kickoff())
