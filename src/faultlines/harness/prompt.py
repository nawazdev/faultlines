"""Task prompt and continuation nudges, following HARNESS_README.md section 5.

Sections 1-5 and 6-7 follow the published text. The "Standard Instructions (0-5)"
wording is not published verbatim, so it is paraphrased from the README's summary.
The three nudge messages are copied from the README's table.
"""

from __future__ import annotations

from .tasks import Task
from .tools import Limits

NUDGE_UNCLOSED_CALL = (
    "Your previous response reached the token limit before the tool call finished closing (<|tool_call|> was "
    "cut off). Do NOT repeat your prior reasoning in thought—emit your next tool call immediately, and if calling "
    "edit_file or write_file, split the change into smaller incremental edits.")
NUDGE_MAX_TOKENS = (
    "Your previous response reached the token limit while thinking before a tool call was completed. Do NOT repeat "
    "your analysis in thought—keep reasoning under a few sentences and emit your next tool call immediately, or call "
    "submit_patch when you have completed and verified your changes.")
NUDGE_CONTINUE = ("Please continue your work using the available tools, or call submit_patch when you have completed "
                  "and verified your changes.")

DEFAULT_SYSTEM = """You are an autonomous software engineer fixing an issue in a Python repository at /workspace.
Explore the code with the tools, find the root cause, make a minimal fix in the library code (not the tests),
check it with a quick targeted command, then call submit_patch as your final action.
Always act through tool calls; keep reasoning short."""


def build_agent_prompt(task: Task, limits: Limits, has_graph: bool, layout: str) -> str:
    parts = [f"You are evaluating a software engineering task for repository {task.repo}.\n\n"
             f"Problem Statement:\n{task.problem_statement}"]
    if task.hints_text and task.hints_text.strip():
        parts.append(f"## Hints:\n{task.hints_text.strip()}")
    budget = ["## Task Budget (Session terminates when any budget is exhausted)",
              f"- Time allowance: {limits.time_minutes:.1f} minutes"]
    if limits.tool_calls is not None:
        budget.append(f"- Tool calls allowance: {limits.tool_calls} calls")
    budget.append(f"- Max loop iterations: {limits.turns} turns")
    parts.append("\n".join(budget))
    parts.append("\n".join([
        "## Execution Environment Rules",
        f"- Single command timeout: {limits.command_timeout} seconds (commands exceeding this fail without ending the session)",
        f"- Command output limit: {limits.max_stdout_chars} characters",
        f"- File view limit: {limits.max_file_lines} lines per read_file call",
        f"- File character limit: {limits.max_file_chars} characters per read_file call",
        "- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY "
        "pre-installed. Do NOT attempt to run pip install or download packages.",
    ]))
    parts.append("\n".join([
        "## Instructions",
        "0. Work strictly inside /workspace.",
        "1. Inspect the relevant code and follow the repository's existing conventions.",
        "2. Make the smallest change in the library code that resolves the issue; do not edit the tests.",
        "3. Prefer quick inline checks (python3 -c \"...\" assertions) over full test sweeps.",
        "4. Put scratch files in /tmp, never in /workspace, so they do not end up in the patch.",
        "5. When done and verified, call submit_patch, then reply with a short final summary.",
    ]))
    if has_graph:
        parts.append("\n".join([
            "## Code Intelligence Tools",
            "This repository has pre-built code graph and embedding data. Use these tools for fast, targeted navigation:",
            "- `search_similar_code(query)`: Find semantically similar functions/classes by keyword.",
            "- `get_code_neighbors(node)`: Find callers, callees, and definitions related to a symbol.",
            "- `get_code_subgraph(nodes)`: Get the induced subgraph for a set of symbols.",
        ]))
    parts.append("## Workspace Layout\n" + layout)
    return "\n\n".join(parts)


def workspace_layout(sandbox) -> str:
    r = sandbox.run("find . -maxdepth 3 -not -path './.git*' -not -name '__pycache__' -not -name '*.pyc' "
                    "| sort | head -150", timeout=30)
    return r.stdout.strip()
