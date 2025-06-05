# Development

## Development Environment

For development, an Anaconda environment is recommended.  This will create a conda environment
called `sdcat`.

```shell
conda env create
conda activate sdcat
```

Alternatively, you can use `poetry` to manage the environment.  This will create a virtual environment
called `sdcat`.

```shell
poetry install
```


## Running pre-commit hooks

Before checking code back in, be sure to run the pre-commit hooks to ensure code quality and consistency.

```shell
poetry run pre-commit run --all-files
```

All tests should run and pass. If you have not installed `pre-commit` yet, you can do so with:

```shell
pip install pre-commit
```

```shell
poetry  pre-commit run --all-file
```
```shell
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Passed
ruff.....................................................................Passed
ruff-format..............................................................Passed
Run pytest before commit.................................................Passed
- hook id: run-pytest
- files were modified by this hook

============================= test session starts ==============================
platform darwin -- Python 3.11.12, pytest-7.4.4, pluggy-1.6.0
rootdir: /Users/dcline/Dropbox/code/ai/sdcat
configfile: pyproject.toml
testpaths: tests
plugins: syrupy-4.6.1, anyio-4.9.0
collected 3 items

tests/test_detect.py ...                                                 [100%]

========================= 3 passed in 65.53s (0:01:05) =========================


```

# Building python package

To build the python package, run the following command:

```shell
poetry build
```

This will create a `dist` directory with the package in it.

Test the package by installing it in a new environment, e.g.:

```shell
pip install dist/mbari-sdcat-0.1.0.tar.gz
```
