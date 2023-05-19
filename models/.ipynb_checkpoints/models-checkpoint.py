import random

import torch
from torch import nn
from einops import rearrange

import numpy as np


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = self.aap(x).view(x.shape[0], -1)

        return x_flat, x


##  Classifier
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


class seq_classifier(nn.Module):
    def __init__(self, configs):
        super(seq_classifier, self).__init__()

        feat_dimension = configs.features_len * configs.final_out_channels

        self.seq_model = nn.Sequential(nn.Linear(feat_dimension, feat_dimension // 4),
                                       nn.ReLU(),
                                       nn.Linear(feat_dimension // 4, feat_dimension // 8)
                                       )

    def forward(self, x):
        predictions = self.seq_model(x)
        return predictions


# Evidential ood detector
class EDL(nn.Module):
    def __init__(self, configs):
        super(EDL, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = self.seq_length * self.num_channels
        # input size: batch_size, 128 channel, 18 seq_length
        self.logits = nn.Linear(self.hid_dim, configs.num_classes)

    def forward(self, x):
        # take the last time step
        predictions = self.logits(x)
        return predictions


class AR_EDL(nn.Module):
    def __init__(self, configs):
        super(AR_EDL, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        # input size: batch_size, 128 channel, 18 seq_length
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)
        self.logits = nn.Linear(configs.AR_hid_dim, configs.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        # take the last time step
        predictions = self.logits(out[:, -1, :]).view(x.size(0), -1)
        return predictions


class Forecastor(nn.Module):
    def __init__(self, configs):
        super(Forecastor, self).__init__()
        features_forecast = configs.features_len - configs.forecast_len
        self.forecasting = nn.Linear(features_forecast, configs.forecast_len)

    def forward(self, x):
        predictions = self.forecasting(x)
        return predictions


class Mask_Predictor(nn.Module):
    def __init__(self, configs):
        super(Mask_Predictor, self).__init__()
        features_forecast = configs.features_len - configs.forecast_len
        self.forecasting = nn.Linear(features_forecast, configs.forecast_len)

    def forward(self, x):
        predictions = self.forecasting(x)
        return predictions


class TemporalVerifier(nn.Module):
    def __init__(self, configs):
        super(TemporalVerifier, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        # input size: batch_size, 128 channel, 18 seq_length
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)
        self.logits = nn.Linear(configs.AR_hid_dim, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        # take the last time step
        predictions = self.logits(out[:, -1, :]).view(x.size(0), -1)
        return predictions

class Mask_predictor(nn.Module):
    def __init__(self, configs):
        super(Mask_predictor, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        # input size: batch_size, 128 channel, 18 seq_length
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)
        # take the last time step
        return out

def masking(x, num_splits=8, num_masked=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask


def shuffling(x, num_splits=10, num_shuffled=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    shuffled_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    full_indices = rand_indices[:, :num_shuffled]

    for i in range(shuffled_patches.shape[1]):
        shuffled_patches[:, i, full_indices[i, :], :] = patches[:, i, full_indices[i, :][torch.randperm(num_shuffled)],
                                                        :]
    shuffled_x = rearrange(shuffled_patches, 'a b p l -> a b (p l)', p=num_splits)

    return x, shuffled_x
