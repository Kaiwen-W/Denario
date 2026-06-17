import os
import re
import json
import functools
from pathlib import Path
import warnings

from .llm import LLM, models


@functools.lru_cache(maxsize=1)
def _model_overrides() -> dict:
    """Optional model-id remap, read from the DENARIO_MODEL_OVERRIDES env var (JSON).

    Lets a custom/OpenAI-compatible endpoint receive model names it accepts (e.g.
    map the dated default ids to bare ones) without editing source. Example:
    DENARIO_MODEL_OVERRIDES='{"gpt-4o-2024-11-20": "gpt-4o"}'
    """
    raw = os.getenv("DENARIO_MODEL_OVERRIDES")
    return json.loads(raw) if raw else {}


def install_cmbagent_model_overrides() -> None:
    """Apply DENARIO_MODEL_OVERRIDES to cmbagent's internal agents too.

    cmbagent builds every agent config via its own ``get_model_config`` and some
    agents (e.g. ``camb_context``, ``plot_judge``) use cmbagent's dated default
    ids that never pass through ``llm_parser``. Wrapping ``get_model_config``
    remaps those as well, so a custom endpoint receives the names it accepts.
    No-op when DENARIO_MODEL_OVERRIDES is unset. Safe to call more than once.
    """
    try:
        import cmbagent.cmbagent as _cc  # patch the bound name cmbagent actually calls
    except Exception:
        return
    orig = getattr(_cc, "get_model_config", None)
    if orig is None or getattr(orig, "_denario_wrapped", False):
        return

    @functools.wraps(orig)
    def wrapped(model, api_keys):
        cfg = orig(model, api_keys)  # api_type inferred from the original name -> stays correct
        cfg["model"] = _model_overrides().get(cfg["model"], cfg["model"])
        return cfg

    wrapped._denario_wrapped = True
    _cc.get_model_config = wrapped

def input_check(str_input: str) -> str:
    """Check if the input is a string with the desired content or the path markdown file, in which case reads it to get the content."""

    if str_input.endswith(".md"):
        with open(str_input, 'r') as f:
            content = f.read()
    elif isinstance(str_input, str):
        content = str_input
    else:
        raise ValueError("Input must be a string or a path to a markdown file.")
    return content

def llm_parser(llm: LLM | str) -> LLM:
    """Get the LLM instance from a string."""

    if isinstance(llm, str):
        try:
            llm = models[llm]
        except KeyError:
            raise KeyError(f"LLM '{llm}' not available. Please select from: {list(models.keys())}")
    new_name = _model_overrides().get(llm.name, llm.name)
    if new_name != llm.name:
        llm = llm.model_copy(update={"name": new_name})  # don't mutate the shared singleton
    return llm

def extract_file_paths(markdown_text):
    """
    Extract the bulleted file paths from markdown text 
    and check if they exist and are absolute paths.
    
    Args:
        markdown_text (str): The markdown text containing file paths
    
    Returns:
        tuple: (existing_paths, missing_paths)
    """
    
    # Pattern to match file paths in markdown bullet points
    pattern = r'-\s*([^\n]+\.(?:csv|txt|md|py|json|yaml|yml|xml|html|css|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|php|rb|pl|sh|bat|sql|log))'
    
    # Find all matches
    matches = re.findall(pattern, markdown_text, re.IGNORECASE)
    
    # Clean up paths and check existence
    existing_paths = []
    missing_paths = []
    
    for match in matches:
        path = match.strip()
        if os.path.exists(path) and os.path.isabs(path):
            existing_paths.append(path)
        else:
            missing_paths.append(path)
    
    return existing_paths, missing_paths

def check_file_paths(content: str) -> None:
    """Check that file paths indicated in content text have the proper format"""

    existing_paths, missing_paths = extract_file_paths(content)

    if len(missing_paths) > 0:
        warnings.warn(
            f"The following data files paths in the data description are not in the right format or do not exist:\n"
            f"{missing_paths}\n"
            f"Please fix them according to the convention '- /absolute/path/to/file.ext'\n"
            f"otherwise this may cause hallucinations in the LLMs."
        )

    if len(existing_paths) == 0:
        warnings.warn(
            "No data files paths were found in the data description. If you want to provide input data, ensure that you indicate their path, otherwise this may cause hallucinations in the LLM in the get_results() workflow later on."
        )

def create_work_dir(work_dir: str | Path, name: str) -> Path:
    """Create working directory"""

    work_dir = os.path.join(work_dir, f"{name}_generation_output")
    os.makedirs(work_dir, exist_ok=True)
    return Path(work_dir)

def get_task_result(chat_history, name: str):
    """Get task result from chat history"""
    
    for obj in chat_history[::-1]:
        if obj['name'] == name:
            result = obj['content']
            break
    task_result = result
    return task_result

def in_notebook():
    """Check whether the code is run from a Jupyter Notebook or not, to use different display options"""
    
    try:
        from IPython import get_ipython # type: ignore
        if 'IPKernelApp' not in get_ipython().config:  # type: ignore # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
