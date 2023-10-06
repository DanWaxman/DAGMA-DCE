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


device = torch.device("cuda:1")
torch.set_default_device(device)

torch.set_default_dtype(torch.double)
utils.set_random_seed(0)
torch.manual_seed(0)

reestimate_graph = False

print('>>> Generating Data <<<')
n, d, s0, graph_type, sem_type = 1000, 10, 20, 'ER', 'gauss'  # 'mlp'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
X = utils.simulate_linear_sem(W_true, n, sem_type)

results_r2_sort_regress = r2_sort_regress(X)
acc_r2_sort_regress = utils.count_accuracy(
    B_true, results_r2_sort_regress != 0)
sid_r2_sort_regress = SID(B_true, results_r2_sort_regress != 0).item()
print('[Var Sort Regress Results] Var-Sortability of X:',
      r2_sortability(X, W_true))
print('[Var Sort Regress Results] SHD:',
      acc_r2_sort_regress['shd'], '| SID:', sid_r2_sort_regress, '| F1:', acc_r2_sort_regress['f1'])

results_var_sort_regress = var_sort_regress(X)
acc_var_sort_regress = utils.count_accuracy(
    B_true, results_var_sort_regress != 0)
sid_var_sort_regress = SID(B_true, results_var_sort_regress != 0).item()
print('[R^2 Sort Regress Results] R^2-Sortability of X:',
      var_sortability(X, W_true))
print('[R^2 Sort Regress Results] SHD:',
      acc_var_sort_regress['shd'], '| SID:', sid_var_sort_regress, '| F1:', acc_var_sort_regress['f1'])

X = torch.from_numpy(X).to(device)

print('\n>>> Performing DAGMA-DCE discovery <<<')
eq_model = nonlinear_dce.DagmaMLP_DCE(dims=[d, 10, 1], bias=True).to(device)
model = nonlinear_dce.DagmaDCE(eq_model)

time_start = time.time()
W_est_dce = model.fit(X, lambda1=0, lambda2=5e-3,
                      lr=2e-4, mu_factor=0.1, mu_init=1, T=4, warm_iter=1*5000, max_iter=1*7000)
time_end = time.time()

W_est_dce_no_thresh = W_est_dce.detach().cpu().numpy()
W_est_dce = abs(W_est_dce_no_thresh) * \
    (abs(W_est_dce_no_thresh) > 0.3)
acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
diff_dce = np.linalg.norm(W_est_dce_no_thresh - abs(W_true))
sid_dce = SID(B_true, W_est_dce != 0).item()

mse_dce = model.mse_loss(model.model(X), X).detach().cpu().numpy()

print('[DAGMA-DCE Results] SHD:', acc_dce['shd'], '| TPR:',
      acc_dce['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dce)
print('[DAGMA-DCE Results] Froebenius Difference from W_true:', diff_dce)
print('[DAGMA-DCE Results] Mean squared error:', mse_dce)
print('[DAGMA-DCE Results] Kendall-Tau:', kendalltau(B_true, W_est_dce))

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
    diff_dagma = np.linalg.norm(W_est_dagma - abs(W_true))
    sid_dagma = SID(B_true, W_est_dagma != 0).item()

    mse_dagma = model.mse_loss(model.model(X), X).detach().cpu().numpy()

    print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
          acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma, '| F1:', acc_dagma['f1'])
    print('[DAGMA Results] Froebenius Difference from W_true:', diff_dagma)
    print('[DAGMA Results] Mean squared error:', mse_dagma)
    print('[DAGMA Results] Kendall-Tau:', kendalltau(B_true, W_est_dagma))

    if reestimate_graph:
        W_est_dagma_no_thresh = eq_model.get_graph(X)[0].detach().cpu().numpy()
        W_est_dagma = W_est_dagma * (W_est_dagma_no_thresh > 0.3)
        acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
        diff_dagma = np.linalg.norm(W_est_dagma - abs(W_true))
        sid_dagma = SID(B_true, W_est_dagma != 0).item()

        mse_dagma = model.mse_loss(model.model(X), X).detach().cpu().numpy()

        print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
              acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma)
        print('[DAGMA Results] Froebenius Difference from W_true:', diff_dagma)
        print('[DAGMA Results] Mean squared error:', mse_dagma)


print('\n>>> Plotting Results <<<')


plt.figure(figsize=(3.5, 3.25))
plt.matshow(W_est_dce_no_thresh - abs(W_true),
            cmap='bwr', vmin=-1, vmax=1, fignum=0)

plt.title('Difference in Estimated \nWeighted Graph with DAGMA-DCE')
plt.xticks([])
plt.yticks([])
plt.colorbar()

offset = 0.48
for i in range(d):
    for j in range(d):
        plt.plot([i-offset, i+offset], [j-offset, j-offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)
        plt.plot([i-offset, i+offset], [j+offset, j+offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)
        plt.plot([i-offset, i-offset], [j-offset, j+offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)
        plt.plot([i+offset, i+offset], [j-offset, j+offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)

plt.tight_layout()
plt.savefig('linear_diff_dce.pdf')

offset = 0.48
plt.figure(figsize=(3.5, 3.25))
plt.matshow(W_est_dagma_no_thresh - abs(W_true),
            cmap='bwr', vmin=-1, vmax=1, fignum=0)

for i in range(d):
    for j in range(d):
        plt.plot([i-offset, i+offset], [j-offset, j-offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)
        plt.plot([i-offset, i+offset], [j+offset, j+offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)
        plt.plot([i-offset, i-offset], [j-offset, j+offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)
        plt.plot([i+offset, i+offset], [j-offset, j+offset],
                 alpha=abs(W_true[j, i]) / np.max(abs(W_true)), color='black', linewidth=2)


plt.title('Difference in Estimated \nWeighted Graph with DAGMA')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig('linear_diff_dagma.pdf')

plt.figure(figsize=(3.5, 3.25))
plt.matshow(abs(W_true), cmap='Greys', vmin=0, vmax=2, fignum=0)

plt.title(r'True Graph $W_\mathrm{true}$')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig('linear_true_graph.pdf')
