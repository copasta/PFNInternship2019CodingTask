import numpy as np

def avg_acc(labels, preds):

    # 平均精度(average accuracy)

    acc_pos = 0
    acc_neg = 0
    for idx in range(len(labels)):
        if labels[idx] == preds[idx] and labels[idx] == 0:
            acc_pos += 1
        elif labels[idx] == preds[idx] and labels[idx] == 1:
            acc_pos += 1
    
    acc_pos = acc_pos / np.sum(labels)
    acc_neg = acc_neg / (len(labels) - np.sum(labels))

    avg_accuracy = (acc_pos + acc_neg) / 2.0
    return avg_accuracy