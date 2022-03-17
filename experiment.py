import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GCNThreeOmics
from processing import split_graph_data, accuracy, feature_selection, pairs_graph
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def do_experiment():
    """step1: load dataset from file,downloaded from TCGA by R script."""
    data = pd.read_csv(filepath_or_buffer='data/' + args.cancer_name + '_all_feature_mutation_CNV.csv', index_col=0)
    x = data.iloc[:, :-1]  # three omics data
    y = data.iloc[:, -1]  # label
    splits = StratifiedShuffleSplit(n_splits=10, random_state=5)  # trainset testset split
    train_index, test_index = next(splits.split(x, y))

    if args.cancer_name == 'BRCA':
        # feature 1
        feature_1_num = 62
        # feature 2
        feature_2_num = 74
    elif args.cancer_name == 'STAD':
        # feature 1
        feature_1_num = 166
        # feature 2
        feature_2_num = 169
    else:
        raise Exception('This cancer type is not Supported.')
    """step2: feature selection for transcript data by HSIC-LASSO"""
    selected_genes = feature_selection(x, y, args.cancer_name)['genes'].to_list()
    # feature 3 transcriptomic

    muta = x.iloc[:, :feature_1_num]
    cnv = x.iloc[:, feature_1_num:feature_1_num + feature_2_num]
    trans = x.iloc[:, feature_1_num + feature_2_num:].filter(items=selected_genes, axis=1)
#   normalization transcriptomic data
    trans1 = StandardScaler().fit_transform(trans)
    trans1 = pd.DataFrame(trans1, index=trans.index, columns=trans.columns)
    data = pd.concat([muta, cnv, trans1, y], axis=1)
    # multi omics
    node_pairs = pairs_graph(trans, args.cancer_name)
    node_pairs = node_pairs[node_pairs.iloc[:, 2] >= args.graph_threshold]
    node_pairs = node_pairs.iloc[:, :2]
    adj, features, labels, idx_train, idx_test = split_graph_data(data, train_index, test_index, node_pairs)
    feature_3_num = features.shape[1] - feature_1_num - feature_2_num
    """step3:GCN model definition,training,test"""
    model = GCNThreeOmics(
        feature_1_num=feature_1_num,
        feature_2_num=feature_2_num,
        feature_3_num=feature_3_num,
        nfeat1=args.feature_1_hid,
        nfeat2=args.feature_2_hid,
        nfeat3=args.feature_3_hid,
        nhid=args.gcn_hidden,
        nclass=labels.max().item() + 1,
        dropout=args.dropout
    )

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda(gpu_id)
        features = features.cuda(gpu_id)
        adj = adj.cuda(gpu_id)
        labels = labels.cuda(gpu_id)
        idx_train = idx_train.cuda(gpu_id)
        idx_test = idx_test.cuda(gpu_id)

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        model.eval()
        if args.verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t), "Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    model.eval()
    output = model(features, adj)
    preds = output[idx_test].max(1)[1].type_as(labels)
    label_one_fold, pred_one_fold = labels[idx_test].cpu().numpy(), preds.cpu().numpy()
    return label_one_fold, pred_one_fold


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_name', action='store_true', default='STAD',
                        help='Cancer data name (BRCA or STAD).')
    parser.add_argument('--graph_threshold', action='store_true', default=0.79,
                        help='Threshold of graph link selection')
    parser.add_argument('--feature_1_hid', action='store_true', default=20,
                        help='Number of hidden features of mutation.')
    parser.add_argument('--feature_2_hid', action='store_true', default=30,
                        help='Number of hidden features of CNV.')
    parser.add_argument('--feature_3_hid', action='store_true', default=65,
                        help='Number of hidden features of transcript.')
    parser.add_argument('--gcn_hidden', default=105,
                        help='Number of hidden units.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.2,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--verbose', default=True,
                        help='print log while model training.')
    gpu_id = 0
    args = parser.parse_args()
    args.verbose = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("model parameter selection ")
    print('graph threshold:', args.graph_threshold, 'feature_1_hidden', args.feature_1_hid, 'feature_2_hidden:',
          args.feature_2_hid, 'feature_3_hidden:', args.feature_3_hid, 'gcn_hidden:', args.gcn_hidden)
    GCN_fs_muta_cnv_res_preds, GCN_fs_muta_cnv_labels = do_experiment()
    print(" Model Prediction Report ")
    print(confusion_matrix(GCN_fs_muta_cnv_labels, GCN_fs_muta_cnv_res_preds))
    print(classification_report(GCN_fs_muta_cnv_labels, GCN_fs_muta_cnv_res_preds, digits=5))
