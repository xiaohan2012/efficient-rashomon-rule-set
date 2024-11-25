# Efficient algorithms to explore the Rashomon set of rule set models

This repository contains the source code of the paper *"Efficient Exploration of the Rashomon Set of Rule Set Models"* (KDD 2024)


# Environment setup

The source code is tested with Python 3.8 on MacOS 14.2.1

``` shell
pip install -r requirements.txt
```


Verify that unit tests pass

``` shell
pytest tests
```

# Example usage

We illustrate the usage of approximate counter and almost-uniform sampler based on synthetic random rules

## Approximate counting

``` python
from bds.utils import generate_random_rules_and_y
from bds.meel import approx_mc2

ub = 0.9  # upper bound on the rule set objective function
lmbd = 0.1  # complexity penalty term

eps = 0.8  # error parameter related to estimation accuracy
delta = 0.8  # the estimation confidence parameter


# generate the input data
random_rules, random_y = generate_random_rules_and_y(
    self.num_pts, self.num_rules, rand_seed=42
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

from bds.utils import generate_random_rules_and_y
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
samples = ug.sample(10, exclude_none=True)
```

# Contact persons

- Han Xiao: xiaohan2012@gmail.com
- Martino Ciaperoni: martino.ciaperoni@aalto.fi


# TODO

- [ ] clean up unused notebooks
- [ ] rename package to `rds`
- [ ] add citation section
