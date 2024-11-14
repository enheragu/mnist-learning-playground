# MNIST dataset Learning Playground

This repository includes a set of MachineLearning/DeepLearning models to be trained with MNIST dataset. It is thought to study the variance differences between models with many training iterations for each of them. The results are stored in yaml files and can be plotted and visualized with a provided script.

## Installation

It is recommended to use a python venv to have dependencies installed, just clone the repository somewhere and execute the following commands to create the venv and install all deps:

```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements
```

## Usage

There are two main scripts in this repository, one is in `src/main.py` with all the training loop, and the other one is `src/plot_distribution.py`. Inside the first script theres a dictionary called `model_iterations`, in that structure the models to be tested can be configured along with the iterations to run for each of them. Feel free to dive deeper in the models if you want, or to modify them at will.