from DagmaDCE import utils, nonlinear, nonlinear_dce
import torch
import time
import numpy as np
from CausalDisco.analytics import r2_sortability, var_sortability
from CausalDisco.baselines import r2_sort_regress, var_sort_regress
from cdt.metrics import SID
import argparse
from notears import nonlinear as nonlinear_notears
from scipy.stats import kendalltau, pearsonr, spearmanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='DagmaDCE Testing',)

    parser.add_argument('-n', '--num_nodes', nargs='+',
                        default=[10, 20, 30, 40, 50], type=int)
    parser.add_argument('-g', '--ER_order', default=4, type=int)
    parser.add_argument('-T', '--num_trials', default=10, type=int)
    parser.add_argument('-s', '--random_seed', default=0, type=int)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-x', '--desc', default='')
    parser.add_argument('-f', '--function_type', default='mlp')
    parser.add_argument('-L', '--loss', default='mse')
    parser.add_argument('-N', '--notears', default=False, type=bool)

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_default_device(device)

    torch.set_default_dtype(torch.double)
    utils.set_random_seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    torch.backends.cudnn.benchmark = True

    use_mse_loss = args.loss == 'mse'

    shd_results = np.zeros((len(args.num_nodes), args.num_trials, 5))
    sid_results = np.zeros((len(args.num_nodes), args.num_trials, 5))
    f1_results = np.zeros((len(args.num_nodes), args.num_trials, 5))
    time_elapsed = np.zeros((len(args.num_nodes), args.num_trials, 3))
    correlations = np.zeros((len(args.num_nodes), args.num_trials, 3))

    for idx_nodes, n_nodes in enumerate(args.num_nodes):
        print('-----------------------------\n' +
              f'| Experiments with {n_nodes} Nodes |\n' +
              '-----------------------------\n')
        for t in range(args.num_trials):
            print(f'Trial {t+1}')

            print('>>> Generating Data <<<')
            n, d, s0, graph_type, sem_type = 1000, n_nodes, n_nodes * \
                args.ER_order, 'ER', args.function_type
            B_true = utils.simulate_dag(d, s0, graph_type)
            X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

            results_r2_sort_regress = r2_sort_regress(X)
            acc_r2_sort_regress = utils.count_accuracy(
                B_true, results_r2_sort_regress != 0)
            sid_r2_sort_regress = SID(
                B_true, results_r2_sort_regress != 0).item()
            print('[Var Sort Regress Results] Var-Sortability of X:',
                  r2_sortability(X, B_true))
            print('[Var Sort Regress Results] SHD:', acc_r2_sort_regress['shd'],
                  '| TPR:', acc_r2_sort_regress['tpr'], '| SID:', sid_r2_sort_regress, '| F1:', acc_r2_sort_regress['f1'])
            shd_results[idx_nodes, t, 0] = acc_r2_sort_regress['shd']
            sid_results[idx_nodes, t, 0] = sid_r2_sort_regress

            results_var_sort_regress = var_sort_regress(X)
            acc_var_sort_regress = utils.count_accuracy(
                B_true, results_var_sort_regress != 0)
            sid_var_sort_regress = SID(
                B_true, results_var_sort_regress != 0).item()
            print('[R^2 Sort Regress Results] R^2-Sortability of X:',
                  var_sortability(X, B_true))
            print('[R^2 Sort Regress Results] SHD:', acc_var_sort_regress['shd'],
                  '| TPR:', acc_var_sort_regress['tpr'], '| SID:', sid_var_sort_regress, '| F1:', acc_var_sort_regress['f1'])
            shd_results[idx_nodes, t, 1] = acc_var_sort_regress['shd']
            sid_results[idx_nodes, t, 1] = sid_var_sort_regress

            acc_empty = utils.count_accuracy(B_true, np.zeros_like(B_true))
            sid_empty = SID(B_true, np.zeros_like(B_true)).item()
            print('[Empty Graph Results] SHD:', acc_empty['shd'], '| TPR:',
                  acc_empty['tpr'], '| SID:', sid_empty, '| F1:', acc_empty['f1'])

            X = torch.from_numpy(X).to(device)

            print('>>> Performing DAGMA-DCE discovery <<<')
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

            W_est_dce = model.fit(X, lambda1=3.5e-2, lambda2=5e-3,
                                  lr=2e-4, mu_factor=0.1, mu_init=0.1, T=4, warm_iter=7000, max_iter=8000)
            time_end = time.time()

            W_est_dce = abs(W_est_dce.detach().cpu().numpy()) * \
                (abs(W_est_dce.detach().cpu().numpy()) > 0.25)
            acc_dce = utils.count_accuracy(B_true, W_est_dce != 0)
            sid_dce = SID(B_true, W_est_dce != 0).item()

            print('[DAGMA-DCE Results] SHD:', acc_dce['shd'], '| TPR:',
                  acc_dce['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dce, '| F1:', acc_dce['f1'])
            shd_results[idx_nodes, t, 2] = acc_dce['shd']
            sid_results[idx_nodes, t, 2] = sid_dce
            f1_results[idx_nodes, t, 2] = acc_dce['f1']
            time_elapsed[idx_nodes, t, 0] = time_end-time_start

            print('>>> Performing DAGMA discovery <<<')
            eq_model = nonlinear.DagmaMLP(
                dims=[d, 10, 1], bias=True, dtype=torch.double).to(device)
            model = nonlinear.DagmaNonlinear(
                eq_model, dtype=torch.double, use_mse_loss=use_mse_loss)

            time_start = time.time()
            W_est_dagma = model.fit(X, lambda1=2e-2, lambda2=0.005,
                                    T=4, lr=2e-4, w_threshold=0.3, mu_init=0.1, warm_iter=70000)
            time_end = time.time()

            acc_dagma = utils.count_accuracy(B_true, W_est_dagma != 0)
            sid_dagma = SID(B_true, W_est_dagma != 0).item()

            print('[DAGMA Results] SHD:', acc_dagma['shd'], '| TPR:',
                  acc_dagma['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_dagma, '| F1:', acc_dagma['f1'])

            shd_results[idx_nodes, t, 3] = acc_dagma['shd']
            sid_results[idx_nodes, t, 3] = sid_dagma
            f1_results[idx_nodes, t, 3] = acc_dagma['f1']
            time_elapsed[idx_nodes, t, 1] = time_end-time_start

            W_est_dagma_flat = W_est_dagma.flatten()
            W_est_dce_flat = W_est_dce.flatten()

            nonzero_idx = np.where(np.logical_or(
                W_est_dagma_flat > 0.0, W_est_dce_flat > 0.0))
            W_est_dagma_nonzero = W_est_dagma_flat[nonzero_idx]
            W_est_dce_nonzero = W_est_dce_flat[nonzero_idx]

            correlations[idx_nodes, t, 0] = pearsonr(
                W_est_dagma_nonzero, W_est_dce_nonzero).statistic
            correlations[idx_nodes, t, 1] = kendalltau(
                W_est_dagma_nonzero, W_est_dce_nonzero).statistic
            correlations[idx_nodes, t, 2] = spearmanr(
                W_est_dagma_nonzero, W_est_dce_nonzero).statistic
            print('[DAGMA-DCE vs DAGMA] Pearson-R:', correlations[idx_nodes, t, 0], '| Kendall Tau:',
                  correlations[idx_nodes, t, 1], '| Spearman-R:', correlations[idx_nodes, t, 2])

            if args.notears:
                print('>>> Performing NOTEARS+ discovery <<<')
                eq_model = nonlinear_notears.NotearsMLP(
                    dims=[d, 10, 1], bias=True).to(device)

                time_start = time.time()
                W_est_notears = nonlinear_notears.notears_nonlinear(
                    eq_model, X, lambda1=2e-2, use_mse_loss=use_mse_loss)
                time_end = time.time()

                acc_notears = utils.count_accuracy(B_true, W_est_notears != 0)
                sid_notears = SID(B_true, W_est_notears != 0).item()

                print('[NOTEARS Results] SHD:', acc_notears['shd'], '| TPR:',
                      acc_notears['tpr'], '| Time Elapsed:', time_end-time_start, '| SID:', sid_notears, '| F1:', acc_notears['f1'])

                shd_results[idx_nodes, t, 4] = acc_notears['shd']
                sid_results[idx_nodes, t, 4] = sid_notears
                f1_results[idx_nodes, t, 4] = acc_notears['f1']
                time_elapsed[idx_nodes, t, 2] = time_end-time_start
            else:
                shd_results[idx_nodes, t, 4] = np.nan
                sid_results[idx_nodes, t, 4] = np.nan
                f1_results[idx_nodes, t, 4] = np.nan
                time_elapsed[idx_nodes, t, 2] = np.nan

    desc = 'nodes_' + '_'.join([str(_) for _ in args.num_nodes]) + '_ER_' + \
        str(args.ER_order) + '_trials_' + \
        str(args.random_seed) + '_l_' + \
        str(args.loss) + '_f_' + \
        str(args.function_type) + '_'.join([args.desc])
    print(f'Saving with description {desc}')
    np.save(f'shd_results_{desc}.npy', shd_results)
    np.save(f'sid_results_{desc}.npy', sid_results)
    np.save(f'f1_results_{desc}.npy', f1_results)
    np.save(f'time_elapsed_{desc}.npy', time_elapsed)
    np.save(f'correlations_{desc}.npy', correlations)
