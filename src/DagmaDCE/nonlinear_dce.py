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
        self.model = model
        self.loss = self.mse_loss if use_mse_loss else self.log_mse_loss

    # def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor):
    #     n, d = target.shape
    #     loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
    #     return loss

    def mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        n, d = target.shape
        return 0.5 / n * torch.sum((output - target) ** 2)

    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(self, max_iter: int, lr: float, lambda1: float, lambda2: float, mu: float, s: float, pbar: tqdm, lr_decay: bool = False, checkpoint: int = 1000, tol: float = 1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(
            0.99, 0.999), weight_decay=mu*lambda2)

        obj_prev = 1e16

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8 if lr_decay else 1.0)

        for i in range(max_iter):
            optimizer.zero_grad()

            if i == 0:
                X_hat = self.model(self.X)
                score = self.loss(X_hat, self.X)
                obj = score

            else:
                W_current, observed_derivs = self.model.get_graph(
                    self.X)

                h_val = self.model.h_func(W_current, s)

                if h_val.item() < 0:
                    return False

                X_hat = self.model(self.X)
                score = self.mse_loss(X_hat, self.X)

                l1_reg = lambda1 * self.model.get_l1_reg(observed_derivs)

                obj = mu * (score + l1_reg) + h_val

            obj.backward()
            optimizer.step()

            if lr_decay and (i+1) % 1000 == 0:
                scheduler.step()

            if i % checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()

                if np.abs((obj_prev - obj_new) / (obj_prev)) <= tol:
                    pbar.update(max_iter - i)
                    break
                obj_prev = obj_new

            pbar.update(1)

        return True

    def fit(self, X, lambda1=0.02, lambda2=0.005, T=4, mu_init=1.0, mu_factor=0.1, s=1.0, warm_iter=5e3, max_iter=8e3, lr=1e-3, w_threshold=0.3, disable_pbar=False):
        mu = mu_init
        self.X = X

        with tqdm(total=(T-1)*warm_iter+max_iter, disable=disable_pbar) as pbar:
            for i in range(int(T)):
                success, s_cur = False, s
                lr_decay = False

                inner_iter = int(max_iter) if i == T-1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)

                while success is False:
                    success = self.minimize(
                        inner_iter, lr, lambda1, lambda2, mu, s_cur, lr_decay=lr_decay, pbar=pbar)

                    if success is False:
                        self.model.load_state_dict(
                            model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break  # lr is too small

                    mu *= mu_factor

        return self.model.get_graph(self.X)[0]


class DagmaMLP_DCE(Dagma_DCE_Module):
    def __init__(self, dims, bias=True, dtype=torch.double):
        torch.set_default_dtype(dtype)

        super(DagmaMLP_DCE, self).__init__()

        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)

        self.fc1 = nn.Linear(self.d, self.d*dims[1], bias=bias)

        # TODO: This might initalize outside M-matrices
        # this hasn't been an issue, but lacks some theoretical
        # guarantees. Should revisit later.
        # nn.init.normal_(self.fc1.weight)
        # nn.init.normal_(self.fc1.bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(
                self.d, dims[l+1], dims[l+2], bias=bias))

        self.fc2 = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)

        x = x.view(-1, self.dims[0], self.dims[1])

        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)

        x = x.squeeze(dim=2)

        return x

    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        x_dummy = x.detach().requires_grad_()
        # import time
        observed_deriv = torch.func.vmap(torch.func.jacrev(
            self.forward))(x_dummy).view(-1, self.d, self.d)

        W = torch.sqrt(torch.mean(observed_deriv ** 2, axis=0).T)

        return W, observed_deriv

    def h_func(self, W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        h = -torch.slogdet(s * self.I - W*W)[1] + self.d * np.log(s)

        return h

    def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))
