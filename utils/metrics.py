import logging
import numpy as np
import torch.nn.functional as F
from sklearn import metrics


def print_metrics_multilabel(y_true, predictions, verbose=True):
    if len(y_true.shape) == 1:
        y_true = F.one_hot(y_true, 10)
        predictions = predictions.softmax(dim=-1)
    else:
        predictions = predictions.sigmoid()
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        logging.info("auc_micro = {:.4f}".format(ave_auc_micro))
        logging.info("auc_macro = {:.4f}".format(ave_auc_macro))
        logging.info("auc_weighted = {:.4f}".format(ave_auc_weighted))

    return {"auc_scores": auc_scores,
            "auroc": ave_auc_micro,
            "auc_macro": ave_auc_macro,
            "auc_weighted": ave_auc_weighted}


def print_metrics_binary(y_true, predictions, verbose=True):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose(
            (1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        # logging.info("confusion matrix:")
        logging.info(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls,
     thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score = 2 * prec1 * rec1 / (prec1 + rec1)
    if verbose:
        logging.info("AUC of ROC = {:.4f}".format(auroc))
        logging.info("AUC of PRC = {:.4f}".format(auprc))
        logging.info("min(+P, Se) = {:.4f}".format(minpse))
        logging.info("f1_score = {:.4f}".format(f1_score))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
        "f1_score": f1_score
    }


def print_metrics_regression(y_true, predictions, verbose=True):
    predictions = np.array(predictions)
    y_true = np.array(y_true)

    mse = metrics.mean_squared_error(y_true, predictions)
    mae = metrics.mean_absolute_error(y_true, predictions)

    if verbose:
        logging.info("MSE = {:.4f}".format(mse))
        logging.info("MAE = {:.4f}".format(mae))

    return {"mse": mse,
            "mae": mae}
    