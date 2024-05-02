import pickle
import random
import numpy as np
from data.dataloader import CustomDataset


class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def load_mimic_iii_mortality(training_ratio=0.8):
    x, y, mask, name = pickle.load(open('./data/MIMIC-III/mortality_normalized.pkl', 'rb'))
    patient_index = list(range(len(x)))
    random.shuffle(patient_index)
    x_len = [len(i) for i in x]

    train_num = int(len(x) * training_ratio)
    val_num = int(len(x) * ((1 - training_ratio) / 2))
    test_num = len(x) - train_num - val_num

    train_data = []
    for idx in patient_index[: train_num]:
        train_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    val_data = []
    for idx in patient_index[train_num : train_num + val_num]:
        val_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    test_data = []
    for idx in patient_index[train_num + val_num :]:
        test_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })

    return CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)


def load_mimic_iii_phenotyping(training_ratio=0.8):
    x, y, mask, name = pickle.load(open('./data/MIMIC-III/phenotyping_normalized.pkl', 'rb'))
    patient_index = list(range(len(x)))
    random.shuffle(patient_index)
    x_len = [len(i) for i in x]

    train_num = int(len(x) * training_ratio)
    val_num = int(len(x) * ((1 - training_ratio) / 2))
    test_num = len(x) - train_num - val_num

    train_data = []
    for idx in patient_index[: train_num]:
        train_data.append({
            "x": x[idx],
            "labels": [float(_) for _ in y[idx]],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    val_data = []
    for idx in patient_index[train_num : train_num + val_num]:
        val_data.append({
            "x": x[idx],
            "labels": [float(_) for _ in y[idx]],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    test_data = []
    for idx in patient_index[train_num + val_num :]:
        test_data.append({
            "x": x[idx],
            "labels": [float(_) for _ in y[idx]],
            "lens": x_len[idx],
            "mask": mask[idx],
        })

    return CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)


def load_mimic_iii_decompensation(training_ratio=0.8):
    x, y, mask, name = pickle.load(open('./data/MIMIC-III/decompensation_normalized.pkl', 'rb'))
    patient_index = list(range(len(x)))
    random.shuffle(patient_index)
    x_len = [len(i) for i in x]

    train_num = int(len(x) * training_ratio)
    val_num = int(len(x) * ((1 - training_ratio) / 2))
    test_num = len(x) - train_num - val_num

    train_data = []
    for idx in patient_index[: train_num]:
        train_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    val_data = []
    for idx in patient_index[train_num : train_num + val_num]:
        val_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    test_data = []
    for idx in patient_index[train_num + val_num :]:
        test_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })

    return CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)


def load_mimic_iii_lengthofstay(training_ratio=0.8):
    x, y, mask, name = pickle.load(open('./data/MIMIC-III/lengthofstay_normalized.pkl', 'rb'))
    patient_index = list(range(len(x)))
    random.shuffle(patient_index)
    x_len = [len(i) for i in x]

    train_num = int(len(x) * training_ratio)
    val_num = int(len(x) * ((1 - training_ratio) / 2))
    test_num = len(x) - train_num - val_num
    
    # y = [get_bin_custom(_, 10) for _ in y]
    y = [[_ / 24] for _ in y]

    train_data = []
    for idx in patient_index[: train_num]:
        train_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    val_data = []
    for idx in patient_index[train_num : train_num + val_num]:
        val_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })
        
    test_data = []
    for idx in patient_index[train_num + val_num :]:
        test_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
        })

    return CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)
