# DAGMA-DCE
Using differential causal effect for interpretable differentiable causal learning.

This repository implements the learning method proposed in "DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery". Based on the learning method proposed in [DAGMA](https://github.com/kevinsbello/dagma), we learn causal graphs in a differentiable manner, with the weighted adjacency matrices having directly interpretable meaning. 

# Citation
If you use any code or results from this project, please consider citing the orignal paper:

```
@article{waxman2024dagma,
  author={Waxman, Daniel and Butler, Kurt and Djuri{\'c}, Petar M},
  journal={IEEE Open Journal of Signal Processing}, 
  title={DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery},
  publisher={IEEE},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={},
  note={Accepted.}}
```

# Installation Instructions 

To install DAGMA-DCE, you can download the git repository and install the package using `pip`:

```
git clone https://github.com/DanWaxman/DAGMA-DCE
cd DAGMA-DCE/src
pip install -e .
```
