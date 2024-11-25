![](https://img.shields.io/badge/ML-Interpretability-lightgreen)  ![](https://img.shields.io/badge/python-3.8-blue)

# Efficient algorithms to explore the Rashomon set of rule set models

This repository contains the source code of the paper *"Efficient Exploration of the Rashomon Set of Rule Set Models"* (KDD 2024)


# Environment setup

The source code is tested against Python 3.8 on MacOS 14.2.1

``` shell
pip install -r requirements.txt
```


Verify that unit tests pass

``` shell
pytest tests
```

# Example usage

We illustrate the usage of approximate counter and almost-uniform sampler applied on synthetic data.

## Preparation

Set up a Ray cluster for parallel computing, e.g.,

``` python
import ray
ray.init()
```

## Approximate counting

``` python
from bds.rule_utils import generate_random_rules_and_y
from bds.meel import approx_mc2

ub = 0.9  # upper bound on the rule set objective function
lmbd = 0.1  # complexity penalty term

eps = 0.8  # error parameter related to estimation accuracy
delta = 0.8  # the estimation confidence parameter


num_pts, num_rules = 100, 10
# generate the input data
random_rules, random_y = generate_random_rules_and_y(
    num_pts, num_rules, rand_seed=42
)

# get an approximate estimation of the number of good rule set models
estimated_count = approx_mc2(
    random_rules,
    random_y,
    lmbd=lmbd,
    ub=ub,
    delta=delta,
    eps=eps,
    rand_seed=42,
    parallel=True,  # using paralle run
)
```

## Almost uniform sampling


``` python
from bds.rule_utils import generate_random_rules_and_y
from bds.meel import UniGen

num_pts, num_rules = 100, 10
random_rules, random_y = generate_random_rules_and_y(
    num_pts, num_rules, rand_seed=42
)

ub = 0.9
eps = 8 #  epsilon parameter that controls the closeness between the sampled distribution and uniform distribution
lmbd = 0.1  # complexity penalty term

sampler = UniGen(random_rules, random_y, lmbd, ub, eps, rand_seed=42)

sampler.prepare()  # collect necessary statistics required for sampling

# sample 10 rule sets almost uniformly from the Rashomon set
samples = sampler.sample(10, exclude_none=True)
```

## Candidate rules extraction on real-world datasets

When working with real-world datasets, the first step is often extract a list of candidate rules.

For this purpose, you may rely on `extract_rules_with_min_support` to extract a list of rules with support above a given threshold.

``` python
import pandas as pd
from bds.candidate_generation import extract_rules_with_min_support

dataset = "compas"
data = pd.read_csv('data/compas_train-binary.csv')  # the features are binary
X = data.to_numpy()[:,:-2]  # extract the feature matrix

attribute_names = list(data.columns[:-2])

candidate_rules = extract_rules_with_min_support(X, attribute_names, min_support=70)

# then you may apply the sampler or count estimator on the candidate rules
```

# Contact persons

- Han Xiao: xiaohan2012@gmail.com
- Martino Ciaperoni: martino.ciaperoni@aalto.fi


# TODO

- [ ] clean up unused notebooks
- [ ] rename package to `rds`
- [ ] add citation section
