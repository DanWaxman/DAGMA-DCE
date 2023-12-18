# Modified from https://github.com/kevinsbello/dagma/blob/main/src/dagma/nonlinear.py
# Modifications Copyright (C) 2023 Dan Waxman

import copy
import torch
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import copy
from tqdm.auto import tqdm
from .locally_connected import LocallyConnected
import abc
import typing


class Dagma_DCE_Module(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def h_func(self, W: torch.Tensor, s: float) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_l1_reg(self, W: torch.Tensor) -> torch.Tensor:
        ...


class DagmaDCE:
    def __init__(self, model: Dagma_DCE_Module, use_mse_loss=True):
        """Initializes a DAGMA DCE model. Requires a `DAGMA_DCE_Module`

        Args:
            model (Dagma_DCE_Module): module implementing adjacency matrix,
                h_func constraint, and L1 regularization
            use_mse_loss (bool, optional): to use MSE loss instead of log MSE loss.
                Defaults to True.
        """
        self.model = model
        self.loss = self.mse_loss if use_mse_loss else self.log_mse_loss

    def mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        """Computes the MSE loss sum (output - target)^2 / (2N)"""
        n, d = target.shape
        return 0.5 / n * torch.sum((output - target) ** 2)

    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        """Computes the MSE loss d / 2 * log [sum (output - target)^2 / N ]"""
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(
        self,
        max_iter: int,
        lr: float,
        lambda1: float,
        lambda2: float,
        mu: float,
        s: float,
        pbar: tqdm,
        lr_decay: bool = False,
        checkpoint: int = 1000,
        tol: float = 1e-3,
    ):
        """Perform minimization using the barrier method optimization

        Args:
            max_iter (int): maximum number of iterations to optimize
            lr (float): learning rate for adam
            lambda1 (float): regularization parameter
            lambda2 (float): weight decay
            mu (float): regularization parameter for barrier method
            s (float): DAMGA constraint hyperparameter
            pbar (tqdm): progress bar to use
            lr_decay (bool, optional): whether or not to use learning rate decay.
                Defaults to False.
            checkpoint (int, optional): how often to checkpoint. Defaults to 1000.
            tol (float, optional): tolerance to terminate learning. Defaults to 1e-3.
        """
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.99, 0.999),
            weight_decay=mu * lambda2,
        )

        obj_prev = 1e16

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8 if lr_decay else 1.0
        )

        for i in range(max_iter):
            optimizer.zero_grad()

            if i == 0:
                X_hat = self.model(self.X)
                score = self.loss(X_hat, self.X)
                obj = score

            else:
                W_current, observed_derivs = self.model.get_graph(self.X)

                h_val = self.model.h_func(W_current, s)

                if h_val.item() < 0:
                    return False

                X_hat = self.model(self.X)
                score = self.mse_loss(X_hat, self.X)

                l1_reg = lambda1 * self.model.get_l1_reg(observed_derivs)

                obj = mu * (score + l1_reg) + h_val

            obj.backward()
            optimizer.step()

            if lr_decay and (i + 1) % 1000 == 0:
                scheduler.step()

            if i % checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()

                if np.abs((obj_prev - obj_new) / (obj_prev)) <= tol:
                    pbar.update(max_iter - i)
                    break
                obj_prev = obj_new

            pbar.update(1)

        return True

    def fit(
        self,
        X: torch.Tensor,
        lambda1: float = 0.02,
        lambda2: float = 0.005,
        T: int = 4,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        s: float = 1.0,
        warm_iter: int = 5e3,
        max_iter: int = 8e3,
        lr: float = 1e-3,
        disable_pbar: bool = False,
    ) -> torch.Tensor:
        """Fits the DAGMA-DCE model

        Args:
            X (torch.Tensor): inputs
            lambda1 (float, optional): regularization parameter. Defaults to 0.02.
            lambda2 (float, optional): weight decay. Defaults to 0.005.
            T (int, optional): number of barrier loops. Defaults to 4.
            mu_init (float, optional): barrier path coefficient. Defaults to 1.0.
            mu_factor (float, optional): decay parameter for mu. Defaults to 0.1.
            s (float, optional): DAGMA constraint hyperparameter. Defaults to 1.0.
            warm_iter (int, optional): number of warmup models. Defaults to 5e3.
            max_iter (int, optional): maximum number of iterations for learning. Defaults to 8e3.
            lr (float, optional): learning rate. Defaults to 1e-3.
            disable_pbar (bool, optional): whether or not to use the progress bar. Defaults to False.

        Returns:
            torch.Tensor: graph returned by the model
        """
        mu = mu_init
        self.X = X

        with tqdm(total=(T - 1) * warm_iter + max_iter, disable=disable_pbar) as pbar:
            for i in range(int(T)):
                success, s_cur = False, s
                lr_decay = False

                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)

                while success is False:
                    success = self.minimize(
                        inner_iter,
                        lr,
                        lambda1,
                        lambda2,
                        mu,
                        s_cur,
                        lr_decay=lr_decay,
                        pbar=pbar,
                    )

                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break  # lr is too small

                    mu *= mu_factor

        return self.model.get_graph(self.X)[0]


class DagmaMLP_DCE(Dagma_DCE_Module):
    def __init__(
        self,
        dims: typing.List[int],
        bias: bool = True,
        dtype: torch.dtype = torch.double,
    ):
        """Initializes the DAGMA DCE MLP module

        Args:
            dims (typing.List[int]): dims
            bias (bool, optional): whether or not to use bias. Defaults to True.
            dtype (torch.dtype, optional): dtype to use. Defaults to torch.double.
        """
        torch.set_default_dtype(dtype)

        super(DagmaMLP_DCE, self).__init__()

        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)

        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)

        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))

        self.fc2 = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sigmoidal feedforward NN

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        x = self.fc1(x)

        x = x.view(-1, self.dims[0], self.dims[1])

        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)

        x = x.squeeze(dim=2)

        return x

    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Get the adjacency matrix defined by the DCE and the batched Jacobian

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor, torch.Tensor: the weighted graph and batched Jacobian
        """
        x_dummy = x.detach().requires_grad_()

        observed_deriv = torch.func.vmap(torch.func.jacrev(self.forward))(x_dummy).view(
            -1, self.d, self.d
        )

        # Adjacency matrix is RMS Jacobian
        W = torch.sqrt(torch.mean(observed_deriv**2, axis=0).T)

        return W, observed_deriv

    def h_func(self, W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        """Calculate the DAGMA constraint function

        Args:
            W (torch.Tensor): adjacency matrix
            s (float, optional): hyperparameter for the DAGMA constraint,
                can be any positive number. Defaults to 1.0.

        Returns:
            torch.Tensor: constraint
        """
        h = -torch.slogdet(s * self.I - W * W)[1] + self.d * np.log(s)

        return h

    def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
        """Gets the L1 regularization

        Args:
            observed_derivs (torch.Tensor): the batched Jacobian matrix

        Returns:
            torch.Tensor: _description_
        """
        return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))
