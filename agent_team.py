"""Optional CrewAI layer. Prompts contain profiles and metrics, never raw rows."""

import json
import os
import urllib.error
import urllib.request
from typing import Any

try:
    from crewai import BaseLLM
except ImportError:
    class BaseLLM:  # type: ignore[no-redef]
        """Import-time fallback; agent execution still checks for CrewAI."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("CrewAI is not installed. Run: pip install -r requirements.txt")


class OllamaCloudLLM(BaseLLM):
    """CrewAI-compatible client for Ollama's authenticated native chat API."""

    provider: str = "ollama-cloud"

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        if tools:
            raise ValueError("This Ollama Cloud adapter does not support tool calling")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        host = (self.base_url or "https://ollama.com").rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]
        if host.endswith("/api"):
            host = host[:-4]
        model_name = self.model.split("/", 1)[-1]
        body = json.dumps(
            {"model": model_name, "messages": messages, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{host}/api/chat",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama Cloud returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama Cloud: {exc.reason}") from exc

        try:
            return payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama Cloud response: {payload}") from exc


def _imports():
    try:
        from crewai import Agent, Crew, LLM, Process, Task
        return Agent, Crew, LLM, Process, Task
    except ImportError as exc:
        raise RuntimeError("CrewAI is not installed. Run: pip install -r requirements.txt") from exc


def _llm():
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is missing from .env")
    return OllamaCloudLLM(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
        api_key=api_key,
    )


def request_architecture(profile: dict[str, Any], results: list[dict[str, Any]], search_space: dict[str, Any]) -> dict[str, Any]:
    Agent, Crew, _, Process, Task = _imports()
    llm = _llm()
    data_analyst = Agent(
        role="Dataset Analyst",
        goal="Summarize the prepared dataset and identify data-quality risks.",
        backstory="You reason only from the supplied derived profile and never request raw rows.",
        llm=llm,
        verbose=False,
    )
    feature_engineer = Agent(
        role="Feature Engineering Adviser",
        goal="Assess the preprocessing and suggest safe feature improvements.",
        backstory="You watch for missing data, categorical encoding and leakage.",
        llm=llm,
        verbose=False,
    )
    baseline_analyst = Agent(
        role="Baseline Analyst",
        goal="Interpret measured baseline results and establish the score to beat.",
        backstory="You rely on validation metrics rather than unsupported claims.",
        llm=llm,
        verbose=False,
    )
    researcher = Agent(
        role="Neural Network Researcher",
        goal="Recommend suitable architecture choices for this classification problem.",
        backstory="You balance model capacity, dataset size and overfitting risk.",
        llm=llm,
        verbose=False,
    )
    optimizer = Agent(
        role="Neural Network Optimizer",
        goal="Choose one valid configuration likely to improve validation accuracy.",
        backstory="You compare measured experiments and remain inside the supplied search space.",
        llm=llm,
        verbose=False,
    )
    experiment_engineer = Agent(
        role="Experiment Engineer",
        goal="Check that the proposed configuration is complete and executable.",
        backstory="You enforce the provided search-space contract exactly.",
        llm=llm,
        verbose=False,
    )

    profile_text = json.dumps(profile, indent=2)
    result_text = json.dumps(results, indent=2)
    space_text = json.dumps(search_space, indent=2)
    data_task = Task(
        description="Review this derived dataset profile. Raw rows are intentionally absent:\n" + profile_text,
        expected_output="A short data-quality summary.",
        agent=data_analyst,
    )
    feature_task = Task(
        description="Assess the preprocessing needs in the dataset profile.",
        expected_output="A short feature and leakage-risk assessment.",
        agent=feature_engineer,
        context=[data_task],
    )
    baseline_task = Task(
        description="Interpret these real validation results:\n" + result_text,
        expected_output="The strongest baseline and the validation score to beat.",
        agent=baseline_analyst,
    )
    research_task = Task(
        description="Recommend a neural-network direction using the profile and earlier assessments.",
        expected_output="A short architecture recommendation.",
        agent=researcher,
        context=[data_task, feature_task, baseline_task],
    )
    optimize_task = Task(
        description=(
            "Choose one configuration from this exact search space:\n" + space_text
            + "\nReturn only JSON with keys: layers, activation, optimizer, learning_rate, batch_size, epochs."
        ),
        expected_output="One valid JSON object and no other text.",
        agent=optimizer,
        context=[baseline_task, research_task],
    )
    check_task = Task(
        description="Check the proposed JSON against the exact search space and return the valid JSON unchanged.",
        expected_output="Only the checked JSON object, without Markdown.",
        agent=experiment_engineer,
        context=[optimize_task],
    )
    crew = Crew(
        agents=[data_analyst, feature_engineer, baseline_analyst, researcher, optimizer, experiment_engineer],
        tasks=[data_task, feature_task, baseline_task, research_task, optimize_task, check_task],
        process=Process.sequential,
        verbose=False,
    )
    output = crew.kickoff()
    text = str(output).strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def create_agent_report(profile: dict[str, Any], final_result: dict[str, Any]) -> str:
    Agent, Crew, _, Process, Task = _imports()
    llm = _llm()
    selector = Agent(
        role="Model Selection Scientist",
        goal="Verify that selection followed the declared validation policy.",
        backstory="You distinguish validation metrics from final test metrics.",
        llm=llm,
        verbose=False,
    )
    critic = Agent(
        role="Experiment Critic",
        goal="Explain measured results accurately and identify limitations.",
        backstory="You are concise and never invent a metric.",
        llm=llm,
        verbose=False,
    )
    reporter = Agent(
        role="Experiment Reporter",
        goal="Write a brief human-readable account of the experiment.",
        backstory="You communicate measured outcomes plainly and preserve stated limitations.",
        llm=llm,
        verbose=False,
    )
    payload = json.dumps({"profile": profile, "result": final_result}, indent=2)
    selection_task = Task(
        description="Verify the model-selection evidence in this payload:\n" + payload,
        expected_output="A short selection-policy check.",
        agent=selector,
    )
    critic_task = Task(
        description="Critique the measured result and identify important limitations.",
        expected_output="A concise evidence-based critique.",
        agent=critic,
        context=[selection_task],
    )
    report_task = Task(
        description="Write at most 120 words summarizing the result in simple language.",
        expected_output="A concise final experiment review.",
        agent=reporter,
        context=[selection_task, critic_task],
    )
    crew = Crew(
        agents=[selector, critic, reporter],
        tasks=[selection_task, critic_task, report_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())
