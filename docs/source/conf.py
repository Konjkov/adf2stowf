import os
import sys

sys.path.insert(0, os.path.abspath('../adf2stowf'))

project = 'adf2stowf'
copyright = '2026, Konkov Vladimir'
release = version = '0.9.1'
author = 'Konkov Vladimir'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

html_static_path = []
