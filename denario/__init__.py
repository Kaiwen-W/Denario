from .denario import Denario, Research, Journal, LLM, models, KeyManager
from .config import REPO_DIR
from .utils import install_cmbagent_model_overrides

# Make DENARIO_MODEL_OVERRIDES also apply to cmbagent's internal-default agents.
install_cmbagent_model_overrides()

__all__ = ['Denario', 'Research', 'Journal', 'REPO_DIR', 'LLM', "models", "KeyManager"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("denario")
except PackageNotFoundError:
    # fallback for editable installs, local runs, etc.
    __version__ = "0.0.0"
