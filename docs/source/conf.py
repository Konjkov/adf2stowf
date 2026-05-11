import os
import sys

sys.path.insert(0, os.path.abspath('../adf2stowf'))

project = 'adf2stowfn'
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

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
