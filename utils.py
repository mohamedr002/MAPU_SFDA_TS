import torch
import torch.nn.functional as F
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime

from skorch import NeuralNetClassifier  # for DIV Risk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type,da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, selected_scenarios, dataset_configs, log_dir, hparams):
    save_dict = {
        "x-domains": selected_scenarios,
        "configs": dataset_configs.__dict__,
        "hparams":  dict(hparams),
        "model_dict": algorithm.state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, "checkpoint.pt")

    torch.save(save_dict, save_path)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100

def _calc_metrics_pretrain(src_pred, src_true,trg_pred, trg_true,  log_dir, home_path, target_names):

    src_pred_labels = np.array(src_pred).astype(int)
    src_true_labels = np.array(src_true).astype(int)
    trg_pred_labels = np.array(trg_pred).astype(int)
    trg_true_labels = np.array(trg_true).astype(int)

    src_rep = classification_report(src_true_labels, src_pred_labels, target_names=target_names, digits=6, output_dict=True)
    trg_rep = classification_report(trg_true_labels, trg_pred_labels, target_names=target_names, digits=6, output_dict=True)

    src_df = pd.DataFrame(src_rep)
    trg_df = pd.DataFrame(trg_rep)

    src_acc = accuracy_score(src_true_labels, src_pred_labels)
    trg_acc = accuracy_score(trg_true_labels, trg_pred_labels)



    # save  repor


    return src_acc * 100, src_df["macro avg"]["f1-score"] * 100, trg_acc *100,  trg_df["macro avg"]["f1-score"] *100

import collections
def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")

def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("algorithms/algorithms.py", os.path.join(destination_dir, "algorithms.py"))
    copy(f"configs/data_model_configs.py", os.path.join(destination_dir, f"data_model_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"configs/sweep_params.py", os.path.join(destination_dir, f"sweep_params.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))



def get_iwcv_value(weight, error):
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    return np.mean(weighted_error)


def get_dev_value(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta

class simple_MLP(nn.Module):
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()

        self.dense0 = nn.Linear(inp_units, inp_units//2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units//2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x   
    
def get_weight_gpu(source_feature, target_feature, validation_feature, configs, device):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    import copy
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = copy.deepcopy(source_feature.detach().cpu()) #source_feature.clone()
    target_feature = copy.deepcopy(target_feature.detach().cpu()) #target_feature.clone()
    source_feature = source_feature.to(device)
    target_feature = target_feature.to(device)
    all_feature = torch.cat((source_feature, target_feature), dim=0)
    all_label = torch.from_numpy(np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)).long()
    
    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    learning_rates = [1e-1, 5e-2, 1e-2]
    val_acc = []
    domain_classifiers = []
    
    for lr in learning_rates:    
        domain_classifier = NeuralNetClassifier(
            simple_MLP,
            module__inp_units = configs.final_out_channels * configs.features_len,
            max_epochs=30,
            lr=lr,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks="disable"
        )
        domain_classifier.fit(feature_for_train.float(), label_for_train.long())
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test.numpy() == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)
    
    index = val_acc.index(max(val_acc))
    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature.to(device).float())
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


def calc_dev_risk(target_model, src_train_dl, tgt_train_dl, src_valid_dl, configs, device):
    src_train_feats, _ = target_model.feature_extractor(src_train_dl.dataset.x_data.float().to(device))
    tgt_train_feats, _ = target_model.feature_extractor(tgt_train_dl.dataset.x_data.float().to(device))
    src_valid_feats, _ = target_model.feature_extractor(src_valid_dl.dataset.x_data.float().to(device))
    src_valid_pred = target_model.classifier(src_valid_feats)

    dev_weights = get_weight_gpu(src_train_feats.to(device), tgt_train_feats.to(device),
                                 src_valid_feats.to(device), configs, device)
    dev_error = F.cross_entropy(src_valid_pred, src_valid_dl.dataset.y_data.long().to(device), reduction='none')
    dev_risk = get_dev_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    # iwcv_risk = get_iwcv_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    return dev_risk


def calculate_risk(target_model, risk_dataloader, device):

    x_data = risk_dataloader.dataset.x_data
    y_data = risk_dataloader.dataset.y_data

    feat, _ = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()

# For DIRT-T
class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]


##### for PCS-FUDA
# MIM
class MomentumSoftmax:
    def __init__(self, num_class, m=1):
        self.softmax_vector = torch.zeros(num_class).detach() + 1.0 / num_class
        self.m = m
        self.num = m

    def update(self, mean_softmax, num=1):
        self.softmax_vector = (
            (self.softmax_vector * self.num) + mean_softmax * num
        ) / (self.num + num)
        self.num += num

    def reset(self):
        # print(self.softmax_vector)
        self.num = self.m


import matplotlib.pyplot as plt
# import umap
# import umap.plot

def _plot_umap(model, src_dl, trg_dl, device, save_dir, hparams, epoch):
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 25}
    plt.rc('font', **font)

    with torch.no_grad():
        # Source flow
        src_data = src_dl.dataset.x_data.float().to(device)
        src_labels = src_dl.dataset.y_data.view((-1)).long()
        src_features = model(src_data)

        # target flow
        trg_data = trg_dl.dataset.x_data.float().to(device)
        trg_labels = trg_dl.dataset.y_data.view((-1)).long()
        if src_dl.dataset.len != trg_dl.dataset.len:
            trg_data = trg_data[:src_dl.dataset.len,:,:]
            trg_labels = trg_labels[:src_dl.dataset.len]


        trg_features = model(trg_data)

        # src-dominant and trg-dominant
        mix_ratio = round(hparams.mix_ratio, 2)
        temporal_shift = hparams.temporal_shift
        src_dominant_data = mix_ratio * src_data + (1 - mix_ratio) * \
                       torch.mean(torch.stack([torch.roll(trg_data, -i, 2) for i in range(temporal_shift)], 2), 2)
        src_dominant_features = model(src_dominant_data)

        trg_dominant_data = mix_ratio * trg_data + (1 - mix_ratio) * \
                       torch.mean(torch.stack([torch.roll(src_data, -i, 2) for i in range(temporal_shift)], 2), 2)
        trg_dominant_features = model(trg_dominant_data)


    if not os.path.exists(os.path.join(save_dir, "umap_plots")):
        os.mkdir(os.path.join(save_dir, "umap_plots"))


    # cmaps = plt.get_cmap('jet')
    src_model_reducer = umap.UMAP(n_neighbors=7, min_dist=0.7, metric='correlation', random_state=42)
    src_embedding = src_model_reducer.fit_transform(src_features.view(src_features.shape[0], -1).detach().cpu().numpy())
    src_dominant_embedding = src_model_reducer.fit_transform(src_dominant_features.detach().cpu().numpy())

    trg_model_reducer = umap.UMAP(n_neighbors=7, min_dist=0.7, metric='correlation', random_state=42)
    trg_embedding = trg_model_reducer.fit_transform(trg_features.view(trg_features.shape[0], -1).detach().cpu().numpy())
    trg_dominant_embedding = trg_model_reducer.fit_transform(trg_dominant_features.detach().cpu().numpy())

    print("Plotting UMAP...")
    plt.rcParams["figure.figsize"] = (16, 10)

    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


    plt.scatter(src_embedding[:, 0], src_embedding[:, 1], c='red', s=30, label="Source", marker='o')
    plt.scatter(trg_embedding[:, 0], trg_embedding[:, 1], c='green', s=30, label="Target", marker='x')

    plt.scatter(src_dominant_embedding[:, 0], src_dominant_embedding[:, 1], c='lightsalmon', s=30, label="Source Dominant", marker='o')
    plt.scatter(trg_dominant_embedding[:, 0], trg_dominant_embedding[:, 1], c='aquamarine', s=30, label="Target Dominant", marker='x')

    plt.legend()

    file_name = f"umap_epoch{epoch}.png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()

