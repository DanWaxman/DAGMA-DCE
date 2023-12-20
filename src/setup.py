from setuptools import setup, find_packages

setup(
    name="DagmaDCE",
    version="0.1",
    url="https://github.com/danwaxman/dagma-dce.git",
    author="Dan Waxman",
    author_email="daniel.waxman1@stonybrook.com",
    description='An implementation of our paper "DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery"',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "igraph",
        "tqdm",
    ],
    license="Apache-2.0",
)
