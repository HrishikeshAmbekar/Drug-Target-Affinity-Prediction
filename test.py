import os
import sys
import torch
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from emetrics import get_aupr, get_cindex, get_rm2, get_ci, get_mse, get_rmse, get_pearson, get_spearman
from utils import *
from scipy import stats
from gnn import GNNNet_2GCN_Layers, GNNNet_2GAT_Layers, GNNNet_3GCN_Layers, GNNNet_3GAT_Layers, GNNNet_1GCN_and_1GAT_Layer
from data_process import create_dataset_for_test


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            # data = data.to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def load_model(model_path):
    model = torch.load(model_path)
    return model


def calculate_metrics(Y, P, dataset='davis', model_st, NUM_EPOCHS ):
    # aupr = get_aupr(Y, P)
    cindex = get_cindex(Y, P)  # DeepDTA
    cindex2 = get_ci(Y, P)  # GraphDTA
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    print('metrics for ', dataset)
    # print('aupr:', aupr)
    print('cindex:', cindex)
    print('cindex2', cindex2)
    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)

    result_file_name = 'results/result_' + model_st + '_' + dataset + '_' + str(NUM_EPOCHS) + '.txt'
    result_str = ''
    result_str += dataset + '\r\n'
    result_str += 'rmse:' + str(rmse) + ' ' + ' mse:' + str(mse) + ' ' + ' pearson:' + str(
        pearson) + ' ' + 'spearman:' + str(spearman) + ' ' + 'ci:' + str(cindex) + ' ' + 'rm2:' + str(rm2)
    print(result_str)
    open(result_file_name, 'w').writelines(result_str)


def plot_density(Y, P, dataset='davis', model_st, NUM_EPOCHS):
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.scatter(P, Y, color='blue', s=40)
    plt.title('density of ' + dataset, fontsize=30, fontweight='bold')
    plt.xlabel('predicted', fontsize=30, fontweight='bold')
    plt.ylabel('measured', fontsize=30, fontweight='bold')
    # plt.xlim(0, 21)
    # plt.ylim(0, 21)
    if dataset == 'davis':
        plt.plot([5, 11], [5, 11], color='black')
    else:
        plt.plot([6, 16], [6, 16], color='black')
    # plt.legend()
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')
    plt.savefig(os.path.join('results', 'plot_'+ model_st+ '_' + dataset + '_' + str(NUM_EPOCHS) + '.png'), dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    dataset = ['davis', 'kiba'][int(sys.argv[1])]  # dataset selection
    print('dataset:', dataset)

    cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]  # gpu selection
    print('cuda_name:', cuda_name)

    TEST_BATCH_SIZE = 512
    models_dir = 'models'
    results_dir = 'results'
    NUM_EPOCHS =int(sys.argv[3])
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    model = [GNNNet_2GCN_Layers(), GNNNet_2GAT_Layers(), GNNNet_3GCN_Layers(), GNNNet_3GAT_Layers(), GNNNet_1GCN_and_1GAT_Layer()][int(sys.argv[4])]
    model_st = [GNNNet_2GCN_Layers.__name__, GNNNet_2GAT_Layers.__name__, GNNNet_3GCN_Layers.__name__, GNNNet_3GAT_Layers.__name__, GNNNet_1GCN_and_1GAT_Layer.__name__][int(sys.argv[4])]
    model.to(device)
    embedding = ['esm1b','esm2','protT5'][int(sys.argv[5])]
    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + embedding +'_' + NUM_EPOCHS + '.model'
    result_file_name = 'results/result_' + model_st + '_' + dataset + '_' + embedding + '_' + NUM_EPOCHS + '.txt'
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_data = create_dataset_for_test(dataset, embedding)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)

    Y, P = predicting(model, device, test_loader)
    calculate_metrics(Y, P, dataset, model_st, NUM_EPOCHS )
    plot_density(Y, P, dataset, model_st, NUM_EPOCHS)
