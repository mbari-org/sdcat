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
poetry shell
```


## Running the tests

Run tests before checking code back in.

```shell
poetry run pytest
```

The tests should run and pass.

```shell
=========================================================================================================================================================================================================================== test session starts ============================================================================================================================================================================================================================
platform darwin -- Python 3.10.13, pytest-7.4.4, pluggy-1.3.0
rootdir: /Users/dcline/Dropbox/code/sdcat
plugins: napari-plugin-engine-0.2.0, anyio-3.7.1, napari-0.4.18, npe2-0.7.3
collected 3 items

tests/test_detect.py ...                                                                                                                                                                                                                                                                                                                                                                                                                                              [100%]

======================================================================================================================================================================================================================= 3 passed in 61.48s (0:01:01) ========================================================================================================================================================================================================================
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
