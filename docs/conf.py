import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# 2. Tell Sphinx the name of your master document (without extension)
root_doc = "index"  # pylint: disable=invalid-name

# 3. Define which file extensions Sphinx should read
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# 4. Point to your references file
bibtex_bibfiles = ["references.bib"]
