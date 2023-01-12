import numpy as np


def generate_targets(gts, num_class):
    targets = np.zeros((len(gts), num_class), dtype=np.float64)
    for item_index, gt in enumerate(gts):
        targets[item_index, gt] = 1

    return targets


def reshape_results(results, num_class):
    num_data = len(results)
    results = [result.reshape(1, num_class) for result in results]
    results = np.concatenate(results, axis=0)
    results = np.reshape(results, (num_data, -1))
    return results


def format_results(results):
    transpose_results = {key_str: [] for key_str in results[0]}
    for item in results:
        for key_str in item:
            transpose_results[key_str].append(item[key_str])

    return transpose_results


def format_and_save(results, num_class, file_path):
    results = reshape_results(results, num_class)
    np.save(file_path, results)


def get_confusion_matrix(num_snr, num_class, snr_info, prs, gts):
    confusion_matrix = np.zeros((num_snr, num_class, num_class), dtype=np.float64)

    for idx in range(prs.shape[0]):
        predict_label = int(np.argmax(prs[idx, :]))
        confusion_matrix[snr_info[idx], gts[idx], predict_label] += 1

    return confusion_matrix


def get_classification_accuracy_with_snr(num_snr, num_class, snr_index, snr_info, prs, gts, prefix=''):
    """Calculate the accuracy with different snr and average snr for evaluation.
    Args:
        num_snr: number of values about snr.
        num_class: size of classification set.
        snr_index: a dict between snr index and snr value, where low snr index stands for the low snr value.
        snr_info: snr index of every test item, where low snr index stands for the low snr value.
        prs: all items' test results with the shape of [N, K].
        gts: all items' true labels with the shape of [N,].
        prefix: the prefix name to log the accuracy results.
    """
    confusion_matrix = get_confusion_matrix(num_snr, num_class, snr_info, prs, gts)

    def _matrix_divide(a, b):
        c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return c

    confusion_matrix = _matrix_divide(confusion_matrix, np.expand_dims(np.sum(confusion_matrix, axis=2), axis=2))

    eval_results = dict()
    for snr in snr_index:
        conf = confusion_matrix[snr_index[snr], :, :]
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        if cor+ncor == 0.0:
            eval_results[prefix + 'snr_{}'.format(snr)] = 0.0
        else:
            eval_results[prefix + 'snr_{}'.format(snr)] = 1.0 * cor/ (cor+ncor)

    conf = np.sum(confusion_matrix, axis=0) / num_snr
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)

    return eval_results


def get_online_confusion_matrix(num_class, prs, gts):
    confusion_matrix = np.zeros((num_class, num_class), dtype=np.float64)

    for idx in range(prs.shape[0]):
        predict_label = int(np.argmax(prs[idx, :]))
        confusion_matrix[gts[idx], predict_label,] += 1

    return confusion_matrix


def get_online_classification_accuracy_for_evaluation(num_class, prs, gts, prefix=''):
    confusion_matrix = get_online_confusion_matrix(num_class, prs, gts)

    cor = np.sum(np.diag(confusion_matrix))
    ncor = np.sum(confusion_matrix) - cor
    eval_results = dict()
    eval_results[prefix + 'snr_mean_all'] = 1.0 * cor / (cor + ncor)

    return eval_results


class Constellation:
    def __init__(self, filter_size=None, filter_stride=None):
        # matrix window info
        self.height_range = [-1, 1]
        self.width_range = [-1, 1]

        # parameters for converting sequence
        # data (2, N) to constellation matrix based on conv mode
        if filter_size is None:
            self.filter_size = [0.05, 0.02]
        else:
            self.filter_size = filter_size
        if filter_stride is None:
            self.filter_stride = [0.05, 0.02]
        else:
            self.filter_stride = filter_stride

    def get_filters(self):
        filters = []
        for filter_size, filter_stride in zip(self.filter_size, self.filter_stride):
            filters.append([filter_size, filter_stride])

        return filters

    def generate_by_filter(self, data):

        constellations = []
        filters = []
        for filter_size, filter_stride in zip(self.filter_size, self.filter_stride):
            matrix_width = int((self.width_range[1] - self.width_range[0] - filter_size) / filter_stride + 1)
            matrix_height = int((self.height_range[1] - self.height_range[0] - filter_size) / filter_stride + 1)

            constellation = np.zeros((matrix_height, matrix_width))

            def axis_is(query_axis_x, query_axis_y):
                axis_x = query_axis_x // filter_stride
                axis_y = query_axis_y // filter_stride
                if axis_x * filter_stride + filter_size < query_axis_x:
                    position = [None, None]
                elif axis_y * filter_stride + filter_size < query_axis_y:
                    position = [None, None]
                else:
                    position = [int(axis_x), int(axis_y)]
                return position

            pos_list = map(axis_is, list(data[0, :]), list(data[1, :]))
            num_point = 0
            for pos in pos_list:
                if pos[0] is not None:
                    constellation[pos[0], pos[1]] += 1
                    num_point += 1
            constellations.append(constellation / num_point)
            filters.append([filter_size, filter_stride])

        return constellations, filters
