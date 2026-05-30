

import importlib.util
from pathlib import Path
import shutil
import warnings

import nami

try:
    from nbformat.validator import MissingIDFieldWarning
except ImportError:
    MissingIDFieldWarning = None
else:
    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)

project = "nami"
author = "Levi Evans"
copyright = "2026, Needle Developers"

version = release = nami.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

if importlib.util.find_spec("myst_nb") is not None:
    extensions.append("myst_nb")
if importlib.util.find_spec("sphinx_copybutton") is not None:
    extensions.append("sphinx_copybutton")
if importlib.util.find_spec("sphinx_autodoc_typehints") is not None:
    extensions.append("sphinx_autodoc_typehints")

if "myst_nb" in extensions:
    source_suffix = {
        ".rst": "restructuredtext",
        ".md": "myst-nb",
        ".ipynb": "myst-nb",
    }
    myst_enable_extensions = [
        "colon_fence",
        "dollarmath",
        "amsmath",
    ]
    myst_dmath_double_inline = True
    myst_heading_anchors = 3
else:
    source_suffix = {
        ".rst": "restructuredtext",
    }

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_generated",
    "**.ipynb_checkpoints",
    "books/test/**",
    "api/summary.md",  # local-review bundle of _generated; not part of the built docs
    ".DS_Store",
]

if importlib.util.find_spec("pydata_sphinx_theme") is not None:
    html_theme = "pydata_sphinx_theme"
else:
    html_theme = "nature"

ROOT = Path(__file__).resolve().parents[1]
books_src = ROOT / "books"
books_dst = ROOT / "docs" / "books"
if books_src.exists():
    if books_dst.exists():
        shutil.rmtree(books_dst)
    shutil.copytree(books_src, books_dst)
nb_execution_mode = "off" 

html_title = f"{project} v{version}"
html_static_path = ["_static"]
html_logo = "assets/nami_logo.svg"
html_favicon = "assets/nami_logo.svg"
html_css_files = ["custom.css"]

if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "github_url": "https://github.com/LeviSamuelEvans/nami",
        "header_links_before_dropdown": 6,
        "logo": {
            #"text": "nami",
            "image_light": "assets/nami_logo.svg",
            "image_dark": "assets/nami_logo.svg",
        },
        "navbar_start": ["navbar-logo"],
        "navbar_center": ["navbar-nav"],
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        "navbar_persistent": ["search-button"],
        "navbar_align": "content",
        "sidebar_includehidden": True,
        "collapse_navigation": True,
        "show_nav_level": 2,
        "show_toc_level": 2,
        "navigation_depth": 4,
        "secondary_sidebar_items": ["page-toc"],
        "show_prev_next": False,
    }
    html_sidebars = {
        "**": ["search-field", "sidebar-nav-bs"],
    }
    html_context = {
        "github_user": "LeviSamuelEvans",
        "github_repo": "nami",
        "github_version": "main",
        "doc_path": "docs",
    }

if "sphinx_copybutton" in extensions:
    copybutton_prompt_text = r">>> |\.\.\. |\$ "
    copybutton_prompt_is_regexp = True

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
