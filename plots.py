import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

import os

# accs_train = [ [] for _ in range(bits_limit+1) ]
# accs_test = [ [] for _ in range(bits_limit+1) ]
# firerate_train = [ [[] for _ in range(n_lif+1)] for _ in range(bits_limit) ]
# firerate_test = [ [[] for _ in range(n_lif+1)] for _ in range(bits_limit) ]
# iters = [ [] for _ in range(bits_limit+1) ]

# accs_train: bits_limit+1, num_runs, num_iters
# accs_test: bits_limit+1, num_runs
# firerate_train: bits_limit, n_lif+1, num_runs, num_iters
# firerate_test: bits_limit, n_lif+1, num_runs
# iters: bits_limit+1, num_runs

# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def plot_accs_train(accs_train, args, ann=False, ExponentialMovingAverage=True):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        accs_train = np.load(os.path.join(data_dir, 'accs_train.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        accs_train = np.array(accs_train)
        np.save(os.path.join(data_dir, 'accs_train.npy'), accs_train)

    if ExponentialMovingAverage:
        accs_train = np.array([ [numpy_ewma_vectorized_v2(accs_train[i][j], 100) for j in range(accs_train.shape[1])] for i in range(accs_train.shape[0]) ])

    fig, ax = plt.subplots(figsize=(30, 15))

    if ann:
        mean_accs = [np.mean(accs_train[i], axis=0) for i in range(bits_limit+1)]
        std_accs = [np.std(accs_train[i], axis=0) for i in range(bits_limit+1)]
        
        ax.plot(mean_accs[0], label='ANN')
        ax.fill_between(range(len(mean_accs[0])), mean_accs[0] - std_accs[0], mean_accs[0] + std_accs[0], alpha=0.3)
        for i in range(1, bits_limit+1):
            ax.plot(mean_accs[i], label=f'{i} bit')
            ax.fill_between(range(len(mean_accs[i])), mean_accs[i] - std_accs[i], mean_accs[i] + std_accs[i], alpha=0.3)
    else:
        mean_accs = [np.mean(accs_train[i], axis=0) for i in range(bits_limit)]
        std_accs = [np.std(accs_train[i], axis=0) for i in range(bits_limit)]

        for i in range(bits_limit):
            ax.plot(mean_accs[i], label=f'{i+1} bit')
            ax.fill_between(range(len(mean_accs[i])), mean_accs[i] - std_accs[i], mean_accs[i] + std_accs[i], alpha=0.3)

    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Multi-bit SNN Training Accuracy')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid()

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_train_acc.pdf'))

def plot_accs_test(accs_test, args, ann=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        accs_test = np.load(os.path.join(data_dir, 'accs_test.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        accs_test = np.array(accs_test)
        np.save(os.path.join(data_dir, 'accs_test.npy'), accs_test)

    fig, ax = plt.subplots(figsize=(30, 15))

    if ann:
        mean_accs = [np.mean(accs_test[i], axis=0) for i in range(bits_limit+1)]
        std_accs = [np.std(accs_test[i], axis=0) for i in range(bits_limit+1)]
        
        ax.plot(mean_accs[0], label='ANN')
        ax.fill_between(range(len(mean_accs[0])), mean_accs[0] - std_accs[0], mean_accs[0] + std_accs[0], alpha=0.3)
        for i in range(1, bits_limit+1):
            ax.plot(mean_accs[i], label=f'{i} bit')
            ax.fill_between(range(len(mean_accs[i])), mean_accs[i] - std_accs[i], mean_accs[i] + std_accs[i], alpha=0.3)
    else:
        mean_accs = [np.mean(accs_test[i], axis=0) for i in range(bits_limit)]
        std_accs = [np.std(accs_test[i], axis=0) for i in range(bits_limit)]

        for i in range(bits_limit):
            ax.plot(mean_accs[i], label=f'{i+1} bit')
            ax.fill_between(range(len(mean_accs[i])), mean_accs[i] - std_accs[i], mean_accs[i] + std_accs[i], alpha=0.3)

    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Testing Accuracy')
    ax.set_title('Multi-bit SNN Testing Accuracy')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid()

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_test_acc.pdf'))

def plot_accs_final(accs_test, args, ann=False, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        accs_test = np.load(os.path.join(data_dir, 'accs_test.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        accs_test = np.array(accs_test)
        np.save(os.path.join(data_dir, 'accs_test.npy'), accs_test)

    accs_test = accs_test[:,:,-1]

    fig, ax = None, None
    if not horizontal:
        fig, ax = plt.subplots(figsize=(2*args.N, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 2*args.N))

    x, labels = None, None
    mean_accs, std_accs = None, None

    if ann:
        mean_accs = [np.mean(accs_test[i]) for i in range(bits_limit+1)]
        std_accs = [np.std(accs_test[i]) for i in range(bits_limit+1)]
        
        x = np.arange(0, bits_limit + 1)
        labels = ["ANN"] + [f'{i+1} bit' for i in range(bits_limit)]

    else:
        mean_accs = [np.mean(accs_test[i]) for i in range(bits_limit)]
        std_accs = [np.std(accs_test[i]) for i in range(bits_limit)]
        
        x = np.arange(0, bits_limit)
        labels = [f'{i+1} bit' for i in range(bits_limit)]

    if not horizontal:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.bar(x, mean_accs, yerr=std_accs, capsize=5)
        ax.set_xlabel('Bitwidth')
        ax.set_ylabel('Test Accuracy')        

        for i, v in enumerate(mean_accs):
            ax.text(i, v + std_accs[i], f'{v*100:.2f}±{std_accs[i]*100:.2f}', ha='center', va='bottom')
    else:
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.barh(x, mean_accs, xerr=std_accs, capsize=5)
        ax.set_ylabel('Bitwidth')
        ax.set_xlabel('Test Accuracy')

        for i, v in enumerate(mean_accs):
            ax.text(v, i+0.1, f'{v*100:.2f}±{std_accs[i]*100:.2f}', ha='center', va='center')
    
    ax.set_title('Multi-bit SNN Test Accuracy')
    ax.grid()
    
    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_final_acc.pdf'))

def plot_accs_quant(accs_test, args, ann=False, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        accs_test = np.load(os.path.join(data_dir, 'accs_test_quant.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        accs_test = np.array(accs_test)
        np.save(os.path.join(data_dir, 'accs_test_quant.npy'), accs_test)

    if not horizontal:
        fig, ax = plt.subplots(figsize=(2*args.N, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 2*args.N))

    x, labels = None, None
    mean_accs, std_accs = None, None

    if ann:
        mean_accs = [np.mean(accs_test[i]) for i in range(bits_limit+1)]
        std_accs = [np.std(accs_test[i]) for i in range(bits_limit+1)]
        
        x = np.arange(0, bits_limit + 1)
        labels = ["ANN"] + [f'{i+1} bit' for i in range(bits_limit)]

    else:
        mean_accs = [np.mean(accs_test[i]) for i in range(bits_limit)]
        std_accs = [np.std(accs_test[i]) for i in range(bits_limit)]
        
        x = np.arange(0, bits_limit)
        labels = [f'{i+1} bit' for i in range(bits_limit)]

    if not horizontal:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.bar(x, mean_accs, yerr=std_accs, capsize=5)
        ax.set_xlabel('Bitwidth')
        ax.set_ylabel('Test Accuracy')

        for i, v in enumerate(mean_accs):
            ax.text(i, v + std_accs[i], f'{v*100:.2f}±{std_accs[i]*100:.2f}', ha='center', va='bottom')
    else:
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.barh(x, mean_accs, xerr=std_accs, capsize=5)
        ax.set_ylabel('Bitwidth')
        ax.set_xlabel('Test Accuracy')

        for i, v in enumerate(mean_accs):
            ax.text(v, i+0.1, f'{v*100:.2f}±{std_accs[i]*100:.2f}', ha='center', va='center')

    ax.set_title('Multi-bit SNN Test Accuracy after Quantization')
    ax.grid()

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_quant_acc.pdf'))

def plot_iters_train(iters, args, ann=False, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        iters = np.load(os.path.join(data_dir, 'iters_train.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        iters = np.array(iters)
        np.save(os.path.join(data_dir, 'iters_train.npy'), iters)

    if not horizontal:
        fig, ax = plt.subplots(figsize=(2*args.N, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 2*args.N))

    x, labels = None, None
    mean_iters, std_iters = None, None

    if ann:
        mean_iters = [np.mean(iters[i]) for i in range(bits_limit+1)]
        std_iters = [np.std(iters[i]) for i in range(bits_limit+1)]
        x = np.arange(0, bits_limit + 1)
        labels = ["ANN"] + [f'{i+1} bit' for i in range(bits_limit)]
    else:
        mean_iters = [np.mean(iters[i]) for i in range(bits_limit)]
        std_iters = [np.std(iters[i]) for i in range(bits_limit)]
        x = np.arange(0, bits_limit)
        labels = [f'{i+1} bit' for i in range(bits_limit)]

    if not horizontal:
        ax.bar(x, mean_iters, yerr=std_iters, capsize=5)
        ax.set_xlabel('Bitwidth')
        ax.set_ylabel('Iterations to Reach Target Training Accuracy')
        ax.set_title('Iterations to Reach Target Training Accuracy for Different Bitwidths')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid()

        for i, v in enumerate(mean_iters):
            ax.text(i, v + std_iters[i], f'{v:.2f}±{std_iters[i]:.2f}', ha='center', va='bottom')
    else:
        ax.barh(x, mean_iters, xerr=std_iters, capsize=5)
        ax.set_ylabel('Bitwidth')
        ax.set_xlabel('Iterations to Reach Target Training Accuracy')
        ax.set_title('Iterations to Reach Target Training Accuracy for Different Bitwidths')
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.grid()

        for i, v in enumerate(mean_iters):
            ax.text(v, i+0.1, f'{v:.2f}±{std_iters[i]:.2f}', ha='center', va='center')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_train_iters.pdf'))

def plot_iters_test(iters, args, ann=False, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        iters = np.load(os.path.join(data_dir, 'iters_test.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        iters = np.array(iters)
        np.save(os.path.join(data_dir, 'iters_test.npy'), iters)

    if not horizontal:
        fig, ax = plt.subplots(figsize=(2*args.N, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 2*args.N))

    x, labels = None, None
    mean_iters, std_iters = None, None

    if ann:
        mean_iters = [np.mean(iters[i]) for i in range(bits_limit+1)]
        std_iters = [np.std(iters[i]) for i in range(bits_limit+1)]
        x = np.arange(0, bits_limit + 1)
        labels = ["ANN"] + [f'{i+1} bit' for i in range(bits_limit)]
    else:
        mean_iters = [np.mean(iters[i]) for i in range(bits_limit)]
        std_iters = [np.std(iters[i]) for i in range(bits_limit)]
        x = np.arange(0, bits_limit)
        labels = [f'{i+1} bit' for i in range(bits_limit)]

    if not horizontal:
        ax.bar(x, mean_iters, yerr=std_iters, capsize=5)
        ax.set_xlabel('Bitwidth')
        ax.set_ylabel('Iterations to Reach Target Testing Accuracy')
        ax.set_title('Iterations to Reach Target Testing Accuracy for Different Bitwidths')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid()

        for i, v in enumerate(mean_iters):
            ax.text(i, v + std_iters[i], f'{v:.2f}±{std_iters[i]:.2f}', ha='center', va='bottom')
    else:
        ax.barh(x, mean_iters, xerr=std_iters, capsize=5)
        ax.set_ylabel('Bitwidth')
        ax.set_xlabel('Iterations to Reach Target Testing Accuracy')
        ax.set_title('Iterations to Reach Target Testing Accuracy for Different Bitwidths')
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.grid()

        for i, v in enumerate(mean_iters):
            ax.text(v, i+0.1, f'{v:.2f}±{std_iters[i]:.2f}', ha='center', va='center')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_test_iters.pdf'))

def plot_firerate_train(firerate_train, args):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        firerate_train = np.load(os.path.join(data_dir, 'firerate_train.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        firerate_train = np.array(firerate_train)
        np.save(os.path.join(data_dir, 'firerate_train.npy'), firerate_train)

    num_spiking = firerate_train.shape[1] - 1

    mean_firerate_train = [ [np.mean(firerate_train[i][j], axis=0) for j in range(num_spiking+1)] for i in range(bits_limit) ]
    std_firerate_train = [ [np.std(firerate_train[i][j], axis=0) for j in range(num_spiking+1)] for i in range(bits_limit) ]

    fig_train, ax_train = plt.subplots(num_spiking+1,1, figsize=(30, 10*(num_spiking+1)))
    for i in range(bits_limit):
        for j in range(num_spiking+1):
            ax_train[j].plot(mean_firerate_train[i][j], label=f'{i+1} bit: {np.mean(mean_firerate_train[i][j]):.2f}')
            ax_train[j].fill_between(range(len(mean_firerate_train[i][j])), mean_firerate_train[i][j] - std_firerate_train[i][j], mean_firerate_train[i][j] + std_firerate_train[i][j], alpha=0.3)

    for j in range(num_spiking+1):
        ax_train[j].set_title(f'nnz{j+1}')
        ax_train[j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_train[j].legend()
        ax_train[j].set_xlabel('Iterations')
        ax_train[j].set_ylabel('Firing Rate')
        ax_train[j].set_ylim(0, 1)
        ax_train[j].grid()

    fig_train.suptitle('Firing Rate for Different Bitwidths during Training')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig_train.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_train_firerate.pdf'))

def plot_firerate_test(firerate_test, args):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        firerate_test = np.load(os.path.join(data_dir, 'firerate_test.npy'))
    else:
        os.makedirs(data_dir, exist_ok=True)
        firerate_test = np.array(firerate_test)
        np.save(os.path.join(data_dir, 'firerate_test.npy'), firerate_test)

    num_spiking = firerate_test.shape[1] - 1

    mean_firerate_test = [ [np.mean(firerate_test[i][j], axis=0) for j in range(num_spiking+1)] for i in range(bits_limit) ]
    std_firerate_test = [ [np.std(firerate_test[i][j], axis=0) for j in range(num_spiking+1)] for i in range(bits_limit) ]

    fig_test, ax_test = plt.subplots(num_spiking+1,1, figsize=(30, 10*(num_spiking+1)))
    for i in range(bits_limit):
        for j in range(num_spiking+1):
            ax_test[j].plot(mean_firerate_test[i][j], label=f'{i+1} bit: {np.mean(mean_firerate_test[i][j]):.2f}')
            ax_test[j].fill_between(range(len(mean_firerate_test[i][j])), mean_firerate_test[i][j] - std_firerate_test[i][j], mean_firerate_test[i][j] + std_firerate_test[i][j], alpha=0.3)

    for j in range(num_spiking+1):
        ax_test[j].set_title(f'nnz{j+1}')
        ax_test[j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_test[j].legend()
        ax_test[j].set_xlabel('Iterations')
        ax_test[j].set_ylabel('Firing Rate')
        ax_test[j].set_ylim(0, 1)
        ax_test[j].grid()

    fig_test.suptitle('Firing Rate for Different Bitwidths during Validation')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig_test.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_test_firerate.pdf'))

def plot_firerate_final(firerate_test, args, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        firerate_test = np.load(os.path.join(data_dir, 'firerate_test.npy'))
    else:
        firerate_test = np.array(firerate_test)

    num_spiking = firerate_test.shape[1] - 1

    dataset_len = firerate_test.shape[-1] // args.epochs

    mean_firerate_test_final = np.empty((bits_limit, num_spiking+1))
    std_firerate_test_final = np.empty((bits_limit, num_spiking+1))
    for i in range(bits_limit):
        for j in range(num_spiking+1):
            mean_firerate_test_final[i][j] = np.mean(firerate_test[i][j][:,-dataset_len:].flatten())
            std_firerate_test_final[i][j] = np.std(firerate_test[i][j][:,-dataset_len:].flatten())

    x = np.arange(0, bits_limit)
    labels = [f'{i+1} bit' for i in range(bits_limit)]

    fig_test, ax_test = None, None
    if not horizontal:
        fig_test, ax_test = plt.subplots(num_spiking+1,1, figsize=(2*args.N, 10*(num_spiking+1)))
    else:
        fig_test, ax_test = plt.subplots(1, num_spiking+1, figsize=(2*args.N*(num_spiking+1), 10))
    for i in range(bits_limit):
        for j in range(num_spiking+1):
            ax_test[j].bar(i, mean_firerate_test_final[i][j], yerr=std_firerate_test_final[i][j], capsize=5, label=f'{i+1} bit: {mean_firerate_test_final[i][j]:.2f}±{std_firerate_test_final[i][j]:.2f}')

    for j in range(num_spiking+1):
        ax_test[j].set_title(f'nnz{j+1}')
        ax_test[j].set_xticks(x)
        ax_test[j].set_xticklabels(labels)
        ax_test[j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_test[j].set_xlabel('Bitwidth')
        ax_test[j].set_ylabel('Firing Rate')
        ax_test[j].set_ylim(0, 1)
        ax_test[j].grid()

        for i, v in enumerate(mean_firerate_test_final[:,j]):
            ax_test[j].text(i, v + std_firerate_test_final[i][j], f'{v:.2f}±{std_firerate_test_final[i][j]:.2f}', ha='center', va='bottom')

    fig_test.suptitle('Final Firing Rate for Different Bitwidths')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig_test.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_final_firerate.pdf'))


def plot_energy_train_gpu(train_iters, args, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        train_iters = np.load(os.path.join(data_dir, 'iters_train.npy'))
    else:
        train_iters = np.array(train_iters)

    energy_train_gpu = np.zeros_like(train_iters)
    for i in range(bits_limit):
        for j in range(args.R):
            energy_train_gpu[i][j] = train_iters[i][j] * args.T[i]

    mean_energy_train_gpu = np.mean(energy_train_gpu, axis=1)
    std_energy_train_gpu = np.std(energy_train_gpu, axis=1)

    if not horizontal:
        fig, ax = plt.subplots(figsize=(2*args.N, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 2*args.N))

    x = np.arange(0, bits_limit)
    labels = [f'{i+1} bit' for i in range(bits_limit)]

    if not horizontal:
        ax.bar(x, mean_energy_train_gpu, yerr=std_energy_train_gpu, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Bitwidth')
        ax.set_ylabel('Energy (GPU)')

        for i, v in enumerate(mean_energy_train_gpu):
            ax.text(i, v + std_energy_train_gpu[i], f'{v:.2f}±{std_energy_train_gpu[i]:.2f}', ha='center', va='bottom')
    else:
        ax.barh(x, mean_energy_train_gpu, xerr=std_energy_train_gpu, capsize=5)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_ylabel('Bitwidth')
        ax.set_xlabel('Energy (GPU)')

        for i, v in enumerate(mean_energy_train_gpu):
            ax.text(v, i+0.1, f'{v:.2f}±{std_energy_train_gpu[i]:.2f}', ha='center', va='center')

    ax.set_title('Energy Consumption Estimation for Training on GPU')
    ax.grid()

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_train_energy_gpu.pdf'))

    relative_energy_train_gpu = energy_train_gpu / mean_energy_train_gpu[0]

    mean_relative_energy_train_gpu = np.mean(relative_energy_train_gpu, axis=1)
    std_relative_energy_train_gpu = np.std(relative_energy_train_gpu, axis=1)

    if not horizontal:
        fig, ax = plt.subplots(figsize=(2*args.N, 15))
    else:
        fig, ax = plt.subplots(figsize=(15, 2*args.N))

    if not horizontal:
        ax.bar(x, mean_relative_energy_train_gpu, yerr=std_relative_energy_train_gpu, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Bitwidth')
        ax.set_ylabel('Relative Energy (GPU)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        for i, v in enumerate(mean_relative_energy_train_gpu):
            ax.text(i, v + std_relative_energy_train_gpu[i], f'{v*100:.2f}±{std_relative_energy_train_gpu[i]*100:.2f}', ha='center', va='bottom')
    else:
        ax.barh(x, mean_relative_energy_train_gpu, xerr=std_relative_energy_train_gpu, capsize=5)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_ylabel('Bitwidth')
        ax.set_xlabel('Relative Energy (GPU)')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))


        for i, v in enumerate(mean_relative_energy_train_gpu):
            ax.text(v, i+0.1, f'{v*100:.2f}±{std_relative_energy_train_gpu[i]*100:.2f}', ha='center', va='center')

    ax.set_title('Normalized Relative Energy Consumption Estimation for Training on GPU')
    ax.grid()

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_train_relative_energy_gpu.pdf'))

def plot_energy_test_nh(firerate_test, args, horizontal=False):
    bits_limit = args.N
    dataset_name = args.dataset

    plots_dir = os.path.join(args.output_dir, args.dataset, 'plots')
    data_dir = os.path.join(args.output_dir, args.dataset, 'data')

    if args.plot_from_data:
        firerate_test = np.load(os.path.join(data_dir, 'firerate_test.npy'))
    else:
        firerate_test = np.array(firerate_test)

    num_spiking = firerate_test.shape[1] - 1

    dataset_len = firerate_test.shape[-1] // args.epochs

    mean_energy_test_nh = np.empty((bits_limit, num_spiking+1))
    std_energy_test_nh = np.empty((bits_limit, num_spiking+1))
    for i in range(bits_limit):
        for j in range(num_spiking+1):
            tmp = firerate_test[i][j][:,-dataset_len:].flatten() * args.T[i]
            mean_energy_test_nh[i][j] = np.mean(tmp)
            std_energy_test_nh[i][j] = np.std(tmp)

    if not horizontal:
        fig_test, ax_test = plt.subplots(num_spiking+1,1, figsize=(2*args.N, 10*(num_spiking+1)))
    else:
        fig_test, ax_test = plt.subplots(1, num_spiking+1, figsize=(2*args.N*(num_spiking+1), 10))

    x = np.arange(0, bits_limit)
    labels = [f'{i+1} bit' for i in range(bits_limit)]

    for i in range(bits_limit):
        for j in range(num_spiking+1):
            ax_test[j].bar(i, mean_energy_test_nh[i][j], yerr=std_energy_test_nh[i][j], capsize=5, label=f'{i+1} bit: {mean_energy_test_nh[i][j]:.2f}±{std_energy_test_nh[i][j]:.2f}')

    for j in range(num_spiking+1):
        ax_test[j].set_title(f'nnz{j+1}')
        ax_test[j].set_xticks(x)
        ax_test[j].set_xticklabels(labels)
        ax_test[j].set_xlabel('Bitwidth')
        ax_test[j].set_ylabel('Energy (NH)')
        ax_test[j].grid()

        for i, v in enumerate(mean_energy_test_nh[:,j]):
            ax_test[j].text(i, v + std_energy_test_nh[i][j], f'{v:.2f}±{std_energy_test_nh[i][j]:.2f}', ha='center', va='bottom')

    fig_test.suptitle('Energy Consumption Estimation for Inference on Neuromorphic Hardware')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig_test.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_test_energy_nh.pdf'))

    mean_relative_energy_test_nh = np.empty((bits_limit, num_spiking+1))
    std_relative_energy_test_nh = np.empty((bits_limit, num_spiking+1))

    for i in range(bits_limit):
        for j in range(num_spiking+1):
            tmp = firerate_test[i][j][:,-dataset_len:].flatten() * args.T[i]
            relative_tmp = tmp / mean_energy_test_nh[0][j]
            mean_relative_energy_test_nh[i][j] = np.mean(relative_tmp)
            std_relative_energy_test_nh[i][j] = np.std(relative_tmp)

    if not horizontal:
        fig, ax = plt.subplots(num_spiking+1, 1, figsize=(2*args.N, 10*(num_spiking+1)))
    else:
        fig, ax = plt.subplots(1, num_spiking+1, figsize=(2*args.N*(num_spiking+1), 10))

    x = np.arange(0, bits_limit)
    labels = [f'{i+1} bit' for i in range(bits_limit)]

    for j in range(num_spiking+1):
        ax[j].bar(x, mean_relative_energy_test_nh[:,j], yerr=std_relative_energy_test_nh[:,j], capsize=5)
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(labels)
        ax[j].set_xlabel('Bitwidth')
        ax[j].set_ylabel('Relative Energy (NH)')
        ax[j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax[j].set_title(f'nnz{j+1}')
        ax[j].grid()

        for i, v in enumerate(mean_relative_energy_test_nh[:,j]):
            ax[j].text(i, v + std_relative_energy_test_nh[i][j], f'{v*100:.2f}±{std_relative_energy_test_nh[i][j]*100:.2f}', ha='center', va='bottom')

    fig.suptitle('Normalized Relative Energy Consumption Estimation for Inference on Neuromorphic Hardware')

    os.makedirs(plots_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{dataset_name.lower()}_test_relative_energy_nh.pdf'))
    
