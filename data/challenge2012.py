import pickle
import random
from data.dataloader import CustomDataset


def load_challenge_2012(training_ratio=0.8):
    x, y, static, mask, name = pickle.load(open('./data/Challenge2012/data_normalized.pkl', 'rb'))
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
            "static": static[idx]
        })
        
    val_data = []
    for idx in patient_index[train_num : train_num + val_num]:
        val_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
            "static": static[idx]
        })
        
    test_data = []
    for idx in patient_index[train_num + val_num :]:
        test_data.append({
            "x": x[idx],
            "labels": y[idx],
            "lens": x_len[idx],
            "mask": mask[idx],
            "static": static[idx]
        })

    return CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)
