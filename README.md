## Installation
1. Clone this repository:
    * ``git clone ...``
2. Install the uv package manager:
    * ``pip install uv``
3. Create a new environment:
    * ``uv venv --python 3.11``
4. Activate the new env:
    * ``source .venv/bin/activate``
5. Install this repository:
    * ``make install``

## Code Quality Hacks
There are a few useful commands in this repository you should probably use.
- `make format` will format all your code using the formatter and linter ruff. This will make both your and our experience better.
- `make check` will check your code for formatting, linting, typing and docstyle. We recommend running this from time to time. It will also be checked when you commit your code.