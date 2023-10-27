# optima_tt


## Description

Numerical experiments for **optima_tt** method from [teneva](https://github.com/AndreiChertkov/teneva) python package. This method finds items which relate to min and max elements of the tensor in the tensor train (TT) format. We use benchmarks from [teneva_bm](https://github.com/AndreiChertkov/teneva_bm) library for computations.


## Installation

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org) (version 3.8);

2. Create a virtual environment:
    ```bash
    conda create --name optima_tt python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate optima_tt
    ```

4. Install dependencies:
    ```bash
    pip install teneva==0.14.8 teneva_bm==0.8.5 seaborn==0.13.0 fpcross==0.5.5
    ```

5. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name optima_tt --all -y
    ```


## Usage

Run `python calc.py random_small`, `python calc.py function_small`, `python calc.py function_big`, `python calc.py random_small_hist`, and `python calc.py fpe`. The results will be presented in the `result` folder.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)
