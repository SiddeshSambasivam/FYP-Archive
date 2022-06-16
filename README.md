# Scientific Machine Learning for Knowledge Discovery

_This repository holds all the work for my final year project. (Jan 2022 - Dec 2022)_

## Update Log

### **Jun**

-   Added the benchmark datasets

    -   AI Feynman
    -   Nguyen (Yet to add this dataset)

-   Added benchmark scripts

-   Added baseline models
    -   Added `gplearn`
    -   Added `Deep symbolic Regression`

## Development Setup

1. [Install miniconda](https://docs.conda.io/en/latest/miniconda.html#:~:text=Miniconda%20is%20a%20free%20minimal,zlib%20and%20a%20few%20others.) (minimal installer for conda; much better and light-weight)

2. Create the development environment

    ```bash
    make env # Creates the conda environment from environment.yml
    ```

3. Activate the conda environment
    ```bash
    conda activate FYP
    ```

## Utility commands

1. To format code using black; run

    ```bash
    make format
    ```

2. To delete all the compiled python files

    ```bash
    make clean
    ```
