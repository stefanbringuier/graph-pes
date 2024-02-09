# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "graph-pes"
copyright = "2023, John Gardner"
author = "John Gardner"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]


templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}
html_theme = "furo"
autodoc_member_order = "bysource"
maximum_signature_line_length = 100
autodoc_typehints = "description"

copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
]
copybutton_selector = "div.copy-button pre"

logo_highlight_colour = "rgb(50, 170, 191)"
code_color = "rgb(50, 170, 191)"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-problematic": code_color,
        "color-brand-primary": logo_highlight_colour,
        "color-brand-content": logo_highlight_colour,
    },
    "dark_css_variables": {
        "color-problematic": code_color,
        "color-brand-primary": logo_highlight_colour,
        "color-brand-content": logo_highlight_colour,
    },
}
