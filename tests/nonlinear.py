from DagmaDCE import utils, nonlinear, nonlinear_dce
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from CausalDisco.analytics import r2_sortability, var_sortability
from CausalDisco.baselines import r2_sort_regress, var_sort_regress
from cdt.metrics import SID
from scipy.stats import kendalltau
import seaborn as sns
sns.set_context("paper")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)

torch.set_default_dtype(torch.double)
utils.set_random_seed(0)
torch.manual_seed(0)

reestimate_graph = False
RESULTS_DIR = ''


#############################
####### Generate Data #######
#############################
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
      acc_r2_sort_regress['shd'], '| SID:', sid_r2_sort_regress, '| F1:', acc_r2_sort_regress['f1'])

results_var_sort_regress = var_sort_regress(X)
acc_var_sort_regress = utils.count_accuracy(
    B_true, results_var_sort_regress != 0)
sid_var_sort_regress = SID(B_true, results_var_sort_regress != 0).item()
print('[R^2 Sort Regress Results] R^2-Sortability of X:',
      var_sortability(X, B_true))
print('[R^2 Sort Regress Results] SHD:',
      acc_var_sort_regress['shd'], '| SID:', sid_var_sort_regress, '| F1:', acc_var_sort_regress['f1'])

X = torch.from_numpy(X).to(device)

##################################
####### Do DAGMA Discovery #######
##################################
print('>>> Performing DAGMA discovery <<<')
eq_model = nonlinear.DagmaMLP(
    dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
model = nonlinear.DagmaNonlinear(eq_model, dtype=torch.double)


time_start = time.time()
W_est_dagma_no_thresh = model.fit(X, lambda1=2e-2, lambda2=5e-3,
                                  lr=2e-3, mu_factor=0.1, mu_init=0.1, w_threshold=0.0)
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

    print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
          acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)
# for lambda1 in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
for lambda1 in [1e-3]:
    print(f'\n>>> Performing DAGMA discovery (lambda1 = {lambda1})<<<')
    eq_model = nonlinear.DagmaMLP(
        dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
    model = nonlinear.DagmaNonlinear(eq_model, dtype=torch.double)

    time_start = time.time()
    W_est_dagma = model.fit(X, lambda1=lambda1, lambda2=5e-3,
                            T=4, w_threshold=0.0)  # 2e-3 performs favorable to 2e-2
    time_end = time.time()

    W_est_dagma_no_thresh = W_est_dagma
    W_est_dagma = W_est_dagma * (W_est_dagma > 0.3)
    acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
    diff_dagma = np.linalg.norm(W_est_dagma - abs(B_true))
    sid_dagma = SID(B_true, W_est_dagma != 0).item()

    mse_dagma = model.mse_loss(model.model(X), X).detach().cpu().numpy()

    print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
          acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma, '| F1:', acc_dagma['f1'])
    print('[DAGMA Results] Froebenius Difference from B_true:', diff_dagma)
    print('[DAGMA Results] Mean squared error:', mse_dagma)
    print('[DAGMA Results] Kendall-Tau:', kendalltau(B_true, W_est_dagma))

    # We can compare with DAGMA results by reestimating the graph under the DAGMA-DCE
    # definition post-hoc. But this isn't a particularly well-posed constrained
    if reestimate_graph:
        W_est_dagma_no_thresh = eq_model.get_graph(X)[0].detach().cpu().numpy()
        W_est_dagma = W_est_dagma * (W_est_dagma_no_thresh > 0.3)
        acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
        diff_dagma = np.linalg.norm(W_est_dagma - abs(B_true))
        sid_dagma = SID(B_true, W_est_dagma != 0).item()

        mse_dagma = model.mse_loss(model.model(X), X).detach().cpu().numpy()

        print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
              acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)
        print('[DAGMA Results] Froebenius Difference from B_true:', diff_dagma)
        print('[DAGMA Results] Mean squared error:', mse_dagma)


######################################
####### Do DAGMA-DCE Discovery #######
######################################
print('\n>>> Performing DAGMA-DCE discovery <<<')
time_start = time.time()

# As noted in the paper, we can pre-train the initial weights using an instance of 
# DAGMA, which is much faster for large graphs
eq_model = nonlinear.DagmaMLP(
    dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
model = nonlinear.DagmaNonlinear(
    eq_model, dtype=torch.double, use_mse_loss=use_mse_loss)

W_est_dagma = model.fit(X, lambda1=2e-2, lambda2=0.005,
                        T=1, lr=2e-4, w_threshold=0.3, mu_init=0.1, warm_iter=70000, max_iter=80000)

# Use DAGMA weights as initial weights for DAGMA-DCE
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

# Kendall's Tau provides a rank correlation between the DAGMA graph and DAGMA-DCE Graph
print('Rank correlation:', kendalltau(W_est_dagma, W_est_dce))

plt.scatter(W_est_dagma_no_thresh.flatten(),
            W_est_dce_no_thresh.detach().cpu().numpy().flatten())
plt.xlabel('DAGMA')
plt.ylabel('DAGMA-DCE')
plt.title('Adjacency Matrix Entries')
plt.savefig(RESULTS_DIR + 'scatter_plot.png')
