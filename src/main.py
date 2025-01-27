import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from ProFed.partitionings import Partitioner

def plot_heatmap(data, labels, areas, name, floating = True):
    sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(data,
                annot=False,
                cmap="BuPu",
                xticklabels=[f'{i}' for i in range(labels)],
                yticklabels=[f'{i}' for i in range(areas)],
                fmt= '.3f' if floating else 'd'
                )
    plt.xlabel('Label')
    plt.ylabel('Area ID')
    if name == 'Dirichlet':
        plt.title('Dirichlet label skewness')
    elif name == 'Hard':
        plt.title('Hard label skewness')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.close()

def to_distribution_matrix(partitioning, training_data, areas):
    matrix = []
    for k, indexes in partitioning.items():
        # print(f'Area {k} has {len(indexes)} images')
        v = [training_data.dataset.targets[index].item() for index in indexes]
        count = Counter(v)
        for i in range(len(training_data.dataset.classes)):
            if i not in count:
                count[i] = 0
        count = dict(sorted(count.items()))
        matrix.append(count)

    rows = [[d[k] for k in d] for d in matrix]
    matrix = np.array(rows)
    return matrix



if __name__ == '__main__':
    matplotlib.rcParams.update({'axes.titlesize': 20})
    matplotlib.rcParams.update({'axes.labelsize': 18})
    matplotlib.rcParams.update({'xtick.labelsize': 15})
    matplotlib.rcParams.update({'ytick.labelsize': 15})
    plt.rcParams.update({"text.usetex": True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    partitioner = Partitioner()
    dataset = partitioner.download_dataset('MNIST')
    training_set, validation_set = partitioner.train_validation_split(dataset, 0.8)

    partitioning_names = ['IID', 'Dirichlet', 'Hard']

    areas = 5

    for name in partitioning_names:
        partitioning = partitioner.partition(name, training_set, areas)
        matrix = to_distribution_matrix(partitioning, training_set, areas)
        plot_heatmap(matrix, len(training_set.dataset.classes), areas, name, False)
