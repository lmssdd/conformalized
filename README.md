# `conformalized`: Scikit-learn compatible Conformalized Regressors

Conformalized Quantile Regressors compatible with Scikit-Learn.
Conformalized Quantile Regression is an algorithm proposed in 2019 by a Romano, Patterson and Càndes, in a self-titled [paper](https://arxiv.org/pdf/1905.03222.pdf).


## Important Links
`scikit-learn` - http://scikit-learn.org/

## Installation
Before installing the module you will need `numpy`, `scipy` and `scikit-learn`.
Dependencies associated with the previous modules may need root privileges to install

```
pip install numpy scipy scikit-learn
```
can also install dependencies with:

```
pip install -r requirements.txt
```

To install `conformalized` execute:
```shell
pip install git+https://github.com/lmssdd/conformalized.git
```

## Usage
It is similar to any other scikit-learn regressor, but only performs quantile regression on specified quantiles, and conformalizes the results if required. A comparison to standard Gradient Boosting is shown in the [notebook](https://github.com/lmssdd/conformalized/blob/master/plot_gradient_boosting_quantile.ipynb)


