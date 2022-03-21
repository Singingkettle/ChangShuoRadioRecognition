import numpy as np


def generate_targets(gts, class_num):
    targets = np.zeros((gts, class_num), dtype=np.float64)
    for item_index, gt in enumerate(gts):
        targets[item_index, gt] = 1

    return targets


def reshape_results(results, class_num):
    data_num = len(results)
    results = [result.reshape(1, class_num) for result in results]
    results = np.concatenate(results, axis=0)
    results = np.reshape(results, (data_num, -1))
    return results


def format_results(results):
    transpose_results = {key_str: [] for key_str in results[0]}
    for item in results:
        for key_str in item:
            transpose_results[key_str].append(item[key_str])

    return transpose_results


def format_and_save(results, class_num, file_path):
    results = reshape_results(results, class_num)
    np.save(file_path, results)


def get_confusion_matrix(snr_num, class_num, snr_info, prs, gts):
    confusion_matrix = np.zeros((snr_num, class_num, class_num), dtype=np.float64)

    for idx in range(prs.shape[0]):
        predict_label = int(np.argmax(prs[idx, :]))
        confusion_matrix[snr_info[idx], gts[idx], predict_label,] += 1

    return confusion_matrix


def get_classification_accuracy_f1(snr_num, class_num, snr_index, snr_info,
                                   prs, gts, prefix='', return_f1=False):
    """Calculate the accuracy and f1 with different snr and average snr.
    Args:
        snr_num: number of values about snr.
        class_num: size of classification set.
        snr_index: a dict between snr index and snr value, where low snr index stands for the low snr value.
        snr_info: snr index of every test item, where low snr index stands for the low snr value.
        prs: all items' test results with the shape of [N, K].
        gts: all items' true labels with the shape of [N,].
        prefix: the prefix name to log the accuracy and f1 results.
        return_f1: whether calculate the f1 score or not.
    """
    confusion_matrix = get_confusion_matrix(snr_num, class_num, snr_info, prs, gts)

    confusion_matrix = confusion_matrix / \
                       np.expand_dims(np.sum(confusion_matrix, axis=2), axis=2)

    eval_results = dict()
    for snr in snr_index:
        conf = confusion_matrix[snr_index[snr], :, :]
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        eval_results[prefix + 'snr_{}'.format(snr)] = 1.0 * cor / (cor + ncor)

    conf = np.sum(confusion_matrix, axis=0) / snr_num
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)

    if return_f1:
        f1s = list()
        for i in range(class_num):
            f1 = 2.0 * conf[i, i] / \
                 (np.sum(conf[i, :]) + np.sum(conf[:, i]))
            f1s.append(f1)
        average_f1 = sum(f1s) / float(class_num)
        return eval_results, average_f1
    else:
        return eval_results
