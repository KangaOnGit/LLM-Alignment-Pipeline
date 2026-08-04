from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.utils.config import PROJECT_ROOT

PROMPT_ROOT = PROJECT_ROOT / "templates" / "prompts"

_env = Environment(
    loader=FileSystemLoader(PROMPT_ROOT),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


def render_prompt(
    template_name: str | Path,
    **context: Any,
) -> str:
    """
    Render a Jinja prompt template.

    Args:
        template_name: Path relative to templates/prompts.
            Example: "system/cot.jinja"
        **context: Variables available to the template.

    Returns:
        The rendered prompt as a string.
    """
    template = _env.get_template(str(template_name))
    return template.render(**context)