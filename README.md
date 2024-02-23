[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/0703344/tree)
[![arXiv](https://img.shields.io/badge/arXiv-2401.02930-b31b1b.svg)](https://arxiv.org/abs/2401.02930)
[![Static Badge](https://img.shields.io/badge/OJSP-2024.3351593-blue?style=flat)](https://doi.org/10.1109/OJSP.2024.3351593) 


# DAGMA-DCE
Using differential causal effect for interpretable differentiable causal discovery.

This repository implements the learning method proposed in "DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery". Based on the learning method proposed in [DAGMA](https://github.com/kevinsbello/dagma), we learn causal graphs in a differentiable manner, with the weighted adjacency matrices having directly interpretable meaning. 

# Citation
The DAGMA-DCE paper is available open-access on [IEEEXplore](https://doi.org/10.1109/OJSP.2024.3351593) and [the arXiv](https://arxiv.org/abs/2401.02930). If you use any code or results from this project, please consider citing the orignal paper:

```
@article{waxman2024dagma,
  author={Waxman, Daniel and Butler, Kurt and Djuri{\'c}, Petar M},
  journal={IEEE Open Journal of Signal Processing}, 
  title={DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery},
  publisher={IEEE},
  year={2024},
  volume={},
  number={},
  pages={1--9},
  doi={10.1109/OJSP.2024.3351593},
}
```

# Installation Instructions 

To install DAGMA-DCE, you can download the git repository and install the package using `pip`:

```
git clone https://github.com/DanWaxman/DAGMA-DCE
cd DAGMA-DCE/src
pip install -e .
```

Alternatively, the code is available on [CodeOcean](https://codeocean.com/capsule/0703344/tree), which provides a full Docker environment to run the code.

# Example

To create an instance of DAGMA-DCE for inference,
```
from DagmaDCE import utils, nonlinear_dce

# Make data
n, d, s0, graph_type, sem_type = 1000, 20, 80, 'ER', 'gp-add'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

# Create a Dagma-DCE module with an MLP
eq_model = nonlinear_dce.DagmaMLP_DCE(
    dims=[d, 10, 1], bias=True).to(device)

# Create a model from the module
model = nonlinear_dce.DagmaDCE(eq_model, use_mse_loss=True)

# Fit the model
W_est = model.fit(X, lambda1=3.5e-2, lambda2=5e-3,
                    lr=2e-4, mu_factor=0.1, mu_init=0.1, 
                    T=4, warm_iter=7000, max_iter=8000)
```

For basic examples, see `tests/linear.py` and `tests/nonlinear.py`. The `tests/linear.py` script reproduces Figure 1 of the paper, and `tests/run_experiment.py` is used to reproduce the results of the paper. 
