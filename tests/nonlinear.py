'''
Output:

Detecting 1 CUDA device(s).
>>> Generating Data <<<
[Var Sort Regress Results] Var-Sortability of X: 0.5124653739612188
[Var Sort Regress Results] SHD: 160 | SID: 468.0
[R^2 Sort Regress Results] R^2-Sortability of X: 0.8421052631578947
[R^2 Sort Regress Results] SHD: 97 | SID: 420.0

>>> Performing DAGMA-DCE discovery <<<
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23000/23000.0 [05:33<00:00, 69.00it/s]
[DAGMA-DCE Results] SHD: 26 | TPR: 0.74 | Time Elapsed: 333.3591310977936 | SID: 224.0
>>> Performing DAGMA discovery <<<
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230000/230000.0 [04:51<00:00, 790.19it/s]
[DAGMA Results] SHD: 27 | TPR: 0.76 | Time Elapsed: 291.0708589553833 | SID: 302.0
'''

from DagmaDCE import utils, nonlinear, nonlinear_dce
import torch
import time
import numpy as np
from CausalDisco.analytics import r2_sortability, var_sortability
from CausalDisco.baselines import r2_sort_regress, var_sort_regress
from cdt.metrics import SID
from cdt.data import load_dataset
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
torch.set_default_device(device)

torch.set_default_dtype(torch.double)
utils.set_random_seed(1)
torch.manual_seed(1)

print('>>> Generating Data <<<')
n, d, s0, graph_type, sem_type = 1000, 20, 80, 'ER', 'gp-add'
B_true = utils.simulate_dag(d, s0, graph_type)
X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
use_mse_loss = True

results_r2_sort_regress = r2_sort_regress(X)
acc_r2_sort_regress = utils.count_accuracy(
    B_true, results_r2_sort_regress != 0)
sid_r2_sort_regress = SID(B_true, results_r2_sort_regress != 0).item()
print('[Var Sort Regress Results] Var-Sortability of X:',
      r2_sortability(X, B_true))
print('[Var Sort Regress Results] SHD:',
      acc_r2_sort_regress['shd'], '| SID:', sid_r2_sort_regress)

results_var_sort_regress = var_sort_regress(X)
acc_var_sort_regress = utils.count_accuracy(
    B_true, results_var_sort_regress != 0)
sid_var_sort_regress = SID(B_true, results_var_sort_regress != 0).item()
print('[R^2 Sort Regress Results] R^2-Sortability of X:',
      var_sortability(X, B_true))
print('[R^2 Sort Regress Results] SHD:',
      acc_var_sort_regress['shd'], '| SID:', sid_var_sort_regress)

X = torch.from_numpy(X).to(device)

print('\n>>> Performing DAGMA-DCE discovery <<<')
# eq_model = nonlinear_dce.DagmaMLP_DCE(
#     dims=[d, 10, 1], bias=True).to(device)
# model = nonlinear_dce.DagmaDCE(eq_model)

# time_start = time.time()
# W_est_dce_no_thresh = model.fit(X, lambda1=2e-2, lambda2=5e-3,
#                                 lr=1e-3, mu_factor=0.1, mu_init=0.1, warm_iter=14000, max_iter=16000)
# time_end = time.time()

# for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.25]:
#     try:
#         W_est_dce = abs(W_est_dce_no_thresh.detach().cpu().numpy()) * \
#             (abs(W_est_dce_no_thresh.detach().cpu().numpy()) > thresh)

#         acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
#     except Exception:
#         W_est_dce = results_var_sort_regress
#         acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
#     sid_dce = SID(B_true, W_est_dce != 0).item()

#     print(f'[DAGMA-DCE Results (Thresh={thresh})] SHD:', acc_dce['shd'], '| TPR:',
#           acc_dce['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dce)

# eq_model = nonlinear_dce.DagmaMLP_DCE(
#     dims=[d, 10, 1], bias=True).to(device)
# model = nonlinear_dce.DagmaDCE(eq_model, use_mse_loss=False)

# time_start = time.time()
# W_est_dce_no_thresh = model.fit(X, lambda1=2e-2, lambda2=5e-3,
#                                 lr=2e-4, mu_factor=0.1, mu_init=0.1, warm_iter=7000, max_iter=8000)
# time_end = time.time()

# for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.25]:
#     try:
#         W_est_dce = abs(W_est_dce_no_thresh.detach().cpu().numpy()) * \
#             (abs(W_est_dce_no_thresh.detach().cpu().numpy()) > thresh)

#         acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
#     except Exception:
#         W_est_dce = results_var_sort_regress
#         acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
#     sid_dce = SID(B_true, W_est_dce != 0).item()

#     print('[DAGMA-DCE Results] SHD:', acc_dce['shd'], '| TPR:',
#           acc_dce['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dce)

print('>>> Performing DAGMA discovery <<<')
# eq_model = nonlinear.DagmaMLP(
#     dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
# model = nonlinear.DagmaNonlinear(eq_model, dtype=torch.double)


# time_start = time.time()
# W_est_dagma_no_thresh = model.fit(X, lambda1=2e-2, lambda2=5e-3,
#                                   lr=2e-3, mu_factor=0.1, mu_init=0.1, w_threshold=0.0)
# time_end = time.time()

# for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
#     try:
#         W_est_dagma = abs(W_est_dagma_no_thresh) * \
#             (abs(W_est_dagma_no_thresh) > thresh)

#         acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
#     except Exception as e:
#         print(e)
#         W_est_dagma = results_var_sort_regress
#         acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
#     sid_dagma = SID(B_true, W_est_dagma != 0).item()

#     print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
#           acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)

eq_model = nonlinear.DagmaMLP(
    dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
model = nonlinear.DagmaNonlinear(
    eq_model, dtype=torch.double, use_mse_loss=False)


time_start = time.time()
W_est_dagma_no_thresh = model.fit(X, lambda1=2e-2, lambda2=5e-3,
                                  lr=2e-4, mu_factor=0.1, mu_init=0.1, w_threshold=0.0,
                                  warm_iter=70000, max_iter=80000)
time_end = time.time()

for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    try:
        W_est_dagma = abs(W_est_dagma_no_thresh) * \
            (abs(W_est_dagma_no_thresh) > thresh)

        acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
    except Exception as e:
        print(e)
        W_est_dagma = results_var_sort_regress
        acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
    sid_dagma = SID(B_true, W_est_dagma != 0).item()

    print(f'[DAGMA-DCE Results (Thresh={thresh})] SHD:', acc_dagma['shd'], '| TPR:',
          acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)

W_est_dagma_2 = (eq_model.get_graph(X)[0]).detach().cpu().numpy()

for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    try:
        W_est_dagma = abs(W_est_dagma_2) * \
            (abs(W_est_dagma_2) > thresh)

        acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
    except Exception as e:
        print(e)
        W_est_dagma = results_var_sort_regress
        acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
    sid_dagma = SID(B_true, W_est_dagma != 0).item()

    print(f'[DAGMA-DCE Results (Thresh={thresh})] SHD:', acc_dagma['shd'], '| TPR:',
          acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)

W_est_dagma_2 = abs(W_est_dagma_2) * (abs(W_est_dagma_2) > 0.25)
# acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
# sid_dagma = SID(B_true, W_est_dagma != 0).item()

# print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
#       acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)

print('Coupling:', kendalltau(W_est_dagma, W_est_dagma_2))

W_est_dagma_comp = W_est_dagma

fc1_weight = eq_model.fc1.weight
fc1_bias = eq_model.fc1.bias
fc2_weight = eq_model.fc2[0].weight
fc2_bias = eq_model.fc2[0].bias

# eq_model = nonlinear.DagmaMLP(
#     dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
# model = nonlinear.DagmaNonlinear(
#     eq_model, dtype=torch.double, use_mse_loss=False)
# fc1_weight = eq_model.fc1.weight
# fc1_bias = eq_model.fc1.bias
# fc2_weight = eq_model.fc2[0].weight
# fc2_bias = eq_model.fc2[0].bias

# time_start = time.time()
# W_est_dagma_no_thresh = model.fit(X, lambda1=2e-2, lambda2=5e-3,
#                                   lr=4e-3, mu_factor=0.1, mu_init=0.1, w_threshold=0.0)
# time_end = time.time()

# for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
#     try:
#         W_est_dagma = abs(W_est_dagma_no_thresh) * \
#             (abs(W_est_dagma_no_thresh) > thresh)

#         acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
#     except Exception as e:
#         print(e)
#         W_est_dagma = results_var_sort_regress
#         acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
#     sid_dagma = SID(B_true, W_est_dagma != 0).item()

#     print(f'[DAGMA-DCE Results (Thresh={thresh})] SHD:', acc_dagma['shd'], '| TPR:',
#           acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)
time_start = time.time()
eq_model = nonlinear.DagmaMLP(
    dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
model = nonlinear.DagmaNonlinear(
    eq_model, dtype=torch.double, use_mse_loss=use_mse_loss)

W_est_dagma = model.fit(X, lambda1=2e-2, lambda2=0.005,
                        T=1, lr=2e-4, w_threshold=0.3, mu_init=0.1, warm_iter=70000, max_iter=80000)
fc1_weight = eq_model.fc1.weight
fc1_bias = eq_model.fc1.bias
fc2_weight = eq_model.fc2[0].weight
fc2_bias = eq_model.fc2[0].bias

eq_model = nonlinear_dce.DagmaMLP_DCE(
    dims=[d, 10, 1], bias=True).to(device)
model = nonlinear_dce.DagmaDCE(eq_model, use_mse_loss=use_mse_loss)
eq_model.fc1.weight = fc1_weight
eq_model.fc1.bias = fc1_bias
eq_model.fc2[0].weight = fc2_weight
eq_model.fc2[0].bias = fc2_bias

W_est_dce_no_thresh = model.fit(X, lambda1=3.5e-2, lambda2=5e-3,
                                lr=2e-4, mu_factor=0.1, mu_init=0.1, T=4, warm_iter=7000, max_iter=8000)
time_end = time.time()

print(torch.sum(torch.abs(eq_model.fc1.weight)))
_, g = eq_model.get_graph(X)
print(eq_model.get_l1_reg(g))
time_end = time.time()

for thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.25]:
    try:
        W_est_dce = abs(W_est_dce_no_thresh.detach().cpu().numpy()) * \
            (abs(W_est_dce_no_thresh.detach().cpu().numpy()) > thresh)

        acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
    except Exception as e:
        print(e)
        W_est_dce = results_var_sort_regress
        acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
    sid_dce = SID(B_true, W_est_dce != 0).item()

    print(f'[DAGMA-DCE Results (Thresh={thresh})] SHD:', acc_dce['shd'], '| TPR:',
          acc_dce['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dce)

print('Coupling:', kendalltau(W_est_dagma_comp, W_est_dce))

plt.scatter(W_est_dagma_no_thresh.flatten(),
            W_est_dce_no_thresh.detach().cpu().numpy().flatten())
plt.savefig('scatter_plot.png')

plt.clf()

plt.hist(W_est_dce_no_thresh.detach().cpu().numpy().flatten())
plt.savefig('hist.png')
