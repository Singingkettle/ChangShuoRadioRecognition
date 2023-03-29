from itertools import combinations
from typing import Union, Dict, Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

try:
    from tsnecuda import TSNE
except ImportError:
    from sklearn.manifold import TSNE
from ..figure_configs import _COLORS, _MARKERS
from ..figures.utils import get_new_fig


class ClassificationMetricsForSingle:
    """Calculate different figures figures for classification,
    which are only legal for the methods of one class for one data."""

    def __init__(self, pps: np.ndarray, gts: np.ndarray, classes: List[str], feas: np.ndarray = None,
                 centers: np.ndarray = None):
        """
        Args:
            pps: all items' prediction probabilities with the shape of [N, C], and sum(pps[i, :]) = 1
            gts: all items' true labels with the shape of [N,], where the value should be between [0, C)
            classes: list with categories' name, and the index of a specific category in the list is corresponding
            to the prediction index.
            feas: all items' feature vectors with the shape of [N, 2]
            centers: centers of different classes with the shape of [C, 2]
        """

        self.pps = pps
        self.gts = gts
        self.feas = feas
        self.centers = centers
        self.num_class = pps.shape[1]
        self.classes = classes
        # tps is true probabilities, which are generated by encoding gts in one-hot manner, and its shape is [N, C]
        self.tps = label_binarize(gts, classes=[i for i in range(self.num_class)])
        self.pair_list = list(combinations([i for i in range(self.num_class)], 2))

        # Ensure the calculation process for a specified figures is only called once
        self._fd = None  # vis fea
        self._cache_fd = False
        self._cm = None
        self._cache_cm = False
        self._as = None
        self._cache_as = False
        self._fs = None
        self._cache_fs = False
        self._pr = None
        self._cache_pr = False
        self._roc = None
        self._cache_roc = False

    def _FeaDistribution(self, feas, centers) -> np.ndarray:

        fig, ax = get_new_fig('Curve', [8, 8])

        points = np.concatenate([feas, centers], axis=0)
        if points.shape[1] > 2:
            points = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(points)
            feas = points[feas.shape[0], :]
            centers = points[centers.shape[0], :]

        x1 = np.round(np.min(points[:, 0]))
        x2 = np.ceil(np.max(points[:, 0]))

        y1 = np.round(np.min(points[:, 1]))
        y2 = np.ceil(np.max(points[:, 1]))

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)

        center_colors = _COLORS[:centers.shape[0]]
        center_markers = _MARKERS[:centers.shape[0]]
        for class_id in range(centers.shape[0]):
            ax.scatter(centers[class_id, 0], centers[class_id, 1], s=2, c=center_colors[class_id],
                       marker=center_markers[class_id], alpha=1, label=self.classes[class_id])

        for class_id in range(centers.shape[0]):
            index = np.where(self.gts == class_id)[0]
            ax.scatter(feas[index[:6], 0], feas[index[:6], 1], s=1,
                       c=center_colors[class_id], m=center_markers[class_id])

        leg = ax.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'})
        leg.get_frame().set_edgecolor('black')
        plt.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        image = np.transpose(image, [2, 0, 1])

        return image

    def _confusion_matrix(self, pps, tps) -> np.ndarray:
        num_class = self.num_class
        confusion_matrix = np.zeros((num_class + 1, num_class + 1), dtype=np.float64)

        for idx in range(pps.shape[0]):
            predict_label = int(np.argmax(pps[idx, :]))
            gt_label = int(np.argmax(tps[idx, :]))
            confusion_matrix[gt_label, predict_label] += 1

        # Calculate precisions of different classes, where the precisions are not normalized
        confusion_matrix[-1, :-1] = np.sum(confusion_matrix[:-1, :-1], axis=0)

        # Calculate recalls of different classes, where the recalls are not normalized
        confusion_matrix[:-1, -1] = np.sum(confusion_matrix[:-1, :-1], axis=1)

        # Calculate number of data
        confusion_matrix[-1, -1] = pps.shape[0]

        return confusion_matrix

    def _accuracy(self, confusion_matrix) -> float:
        cor = np.sum(np.diag(confusion_matrix[:-1, :-1]))
        accuracy = 1.0 * cor / confusion_matrix[-1, -1]

        return accuracy

    def _precision_recall(self, pps, tps) -> Dict[str, Dict[Union[Any, str], np.ndarray]]:
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(self.num_class):
            precision[i], recall[i], _ = precision_recall_curve(tps[:, i], pps[:, i])
            average_precision[i] = average_precision_score(tps[:, i], pps[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision['micro'], recall["micro"], _ = precision_recall_curve(tps.ravel(), pps.ravel())
        average_precision["micro"] = average_precision_score(tps.ravel(), pps.ravel(), average="micro")

        pr = dict(precision=precision, recall=recall, average_precision=average_precision)

        return pr

    def _f1_score(self, confusion_matrix) -> Dict[str, float]:
        tp_fn = confusion_matrix[:-1, -1]
        tp_fp = confusion_matrix[-1, :-1]
        f1_score = 2 * np.diag(confusion_matrix[:-1, :-1]) / (tp_fn + tp_fp)
        f1_score = {self.classes[index]: f1 for index, f1 in enumerate(f1_score)}
        f1_score['AllClass'] = 1.0 * sum(f1_score.values()) / len(f1_score)

        return f1_score

    def _roc_ovr(self, pps, tps) -> Dict[str, Dict[str, np.ndarray]]:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_class):
            fpr[i], tpr[i], _ = roc_curve(tps[:, i], pps[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # ROC curve using micro-averaged OvR
        fpr["micro"], tpr["micro"], _ = roc_curve(tps.ravel(), pps.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # ROC curve using the OvR macro-average
        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(self.num_class):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= self.num_class

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc = dict(fpr=fpr, tpr=tpr, auc=roc_auc)

        return roc

    def _roc_ovo(self, pps, gts) -> Dict[str, Dict[Tuple[int, int], Dict[Any, Any]]]:

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        for ix, (label_a, label_b) in enumerate(self.pair_list):
            fpr[(label_a, label_b)] = dict()
            tpr[(label_a, label_b)] = dict()
            roc_auc[(label_a, label_b)] = dict()

            a_mask = gts == label_a
            b_mask = gts == label_b
            ab_mask = np.logical_or(a_mask, b_mask)

            a_true = a_mask[ab_mask]
            b_true = b_mask[ab_mask]

            fpr_a, tpr_a, _ = roc_curve(a_true, pps[ab_mask, label_a])
            fpr_b, tpr_b, _ = roc_curve(b_true, pps[ab_mask, label_b])

            auc_a = auc(fpr_a, tpr_a)
            auc_b = auc(fpr_b, tpr_b)

            fpr[(label_a, label_b)][label_a] = fpr_a
            fpr[(label_a, label_b)][label_b] = fpr_b
            tpr[(label_a, label_b)][label_a] = tpr_a
            tpr[(label_a, label_b)][label_b] = tpr_b
            roc_auc[(label_a, label_b)][label_a] = auc_a
            roc_auc[(label_a, label_b)][label_b] = auc_b

            mean_tpr = (np.interp(fpr_grid, fpr_a, tpr_a) + np.interp(fpr_grid, fpr_b, tpr_b)) / 2
            mean_auc = auc(fpr_grid, mean_tpr)

            fpr[(label_a, label_b)]['macro'] = fpr_grid
            tpr[(label_a, label_b)]['macro'] = mean_tpr
            roc_auc[(label_a, label_b)]['macro'] = mean_auc

        roc = dict(fpr=fpr, tpr=tpr, auc=roc_auc)

        return roc

    @property
    def FeaDistribution(self) -> np.ndarray:
        if not self._cache_fd:
            image = self._FeaDistribution(self.feas, self.centers)
            self._fd = image
            self._cache_fd = True

        return self._fd

    @property
    def confusion_matrix(self) -> np.ndarray:
        if not self._cache_cm:
            confusion_matrix = self._confusion_matrix(self.pps, self.tps)
            self._cm = confusion_matrix
            self._cache_cm = True

        return self._cm

    @property
    def ACC(self) -> float:
        if not self._cache_as:
            accuracy = self._accuracy(self.confusion_matrix)
            self._as = accuracy
            self._cache_as = True
        return self._as

    @property
    def F1(self) -> Dict[str, float]:
        if not self._cache_fs:
            f1_score = self._f1_score(self.confusion_matrix)
            self._fs = f1_score
            self._cache_fs = True

        return self._fs

    @property
    def precision_recall(self) -> Dict[str, Dict[Union[Any, str], np.ndarray]]:
        if not self._cache_pr:
            pr = self._precision_recall(self.pps, self.gts)
            self._pr = pr
            self._cache_pr = True

        return self._pr

    @property
    def roc(self) -> Dict[str, Union[Dict[str, Dict[str, ndarray]], Dict[str, Dict[Tuple[int, int], Dict[Any, Any]]]]]:
        if not self._cache_roc:
            roc = dict()
            roc['ovr'] = self._roc_ovr(self.pps, self.gts)
            roc['ovo'] = self._roc_ovo(self.pps, self.gts)
            self._roc = roc
            self._cache_roc = True

        return self._roc


class ClassificationMetricsWithSNRForSingle(ClassificationMetricsForSingle):
    def __init__(self, pps: np.ndarray, gts: np.ndarray, snrs: np.ndarray, classes: List[str],
                 feas: np.ndarray = None, centers: np.ndarray = None):
        """
        Args:
            pps (np.ndarray): all items' prediction probabilities with the shape of [N, C], and sum(pps[i, :]) = 1
            gts (np.ndarray): all items' true labels with the shape of [N,], where the value should be between [0, C)
            snrs (np.ndarray): all items' snr with the shape of [N,]
            classes (List[str]): list with categories' name, and the index of a specific category in the list is
            corresponding to the prediction index.
            feas: all items' feature vectors with the shape of [N, 2]
            centers: centers of different classes with the shape of [C, 2]
        """
        super(ClassificationMetricsWithSNRForSingle, self).__init__(pps, gts, classes, feas=feas, centers=centers)
        self.snrs = snrs
        self.snr_set = np.sort(np.unique(snrs))
        snr_to_index = {snr: index for index, snr in enumerate(self.snr_set)}
        pps_along_snr = {snr: self.pps[snrs == snr, :] for snr in snr_to_index}
        tps_along_snr = {snr: self.tps[snrs == snr, :] for snr in snr_to_index}
        gts_along_snr = {snr: self.gts[snrs == snr] for snr in snr_to_index}
        self.pps_along_snr = pps_along_snr
        self.tps_along_snr = tps_along_snr
        self.gts_along_snr = gts_along_snr

        self.num_snr = len(snr_to_index)
        self.snr_to_index = snr_to_index

    def _FeaDistribution(self, feas, centers, snrs) -> np.ndarray:

        fig, ax = get_new_fig('Curve', [8, 8])

        points = np.concatenate([feas, centers], axis=0)
        if points.shape[1] > 2:
            points = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(points)
            feas = points[:feas.shape[0], :]
            centers = points[feas.shape[0]:, :]

        x1 = np.round(np.min(points[:, 0]))
        x2 = np.ceil(np.max(points[:, 0]))

        y1 = np.round(np.min(points[:, 1]))
        y2 = np.ceil(np.max(points[:, 1]))

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)

        center_colors = _COLORS[:centers.shape[0]]
        center_markers = _MARKERS[:centers.shape[0]]

        for class_id in range(centers.shape[0]):
            ax.scatter(centers[class_id, 0], centers[class_id, 1], s=30, c=center_colors[class_id],
                       marker=center_markers[class_id], alpha=1, label=self.classes[class_id])

        alpha_size = 0.9
        dalpha = alpha_size / len(self.snr_set)
        snr_dict = {self.snr_set[i]: i for i in range(len(self.snr_set))}

        for snr in self.snr_set:
            for class_id in range(centers.shape[0]):
                index = np.where((self.snrs == snr) & (self.gts == class_id))[0]
                ax.scatter(feas[index[:6], 0], feas[index[:6], 1], s=15, c=center_colors[class_id],
                           marker=center_markers[class_id], alpha=snr_dict[snr] * dalpha + 1 - alpha_size)

        leg = ax.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'})
        leg.get_frame().set_edgecolor('black')

        plt.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        image = np.transpose(image, [2, 0, 1])
        return image

    @property
    def FeaDistribution(self) -> Dict[str, np.ndarray]:
        if not self._cache_fd:
            image = self._FeaDistribution(self.feas, self.centers, self.snrs)
            self._fd = {'Image/FeaDistribution': image}
            self._cache_fd = True

        return self._fd

    @property
    def confusion_matrix(self) -> Dict[str, np.ndarray]:
        if not self._cache_cm:
            confusion_matrix = dict()
            for snr in self.snr_to_index:
                confusion_matrix[f'{snr:02d}dB'] = self._confusion_matrix(self.pps_along_snr[snr],
                                                                          self.tps_along_snr[snr])
            # Add All SNR together
            confusion_matrix['AllSNR'] = self._confusion_matrix(self.pps, self.tps)

            self._cm = confusion_matrix
            self._cache_cm = True

        return self._cm

    @property
    def ACC(self) -> Dict[Union[str, Any], float]:
        if not self._cache_as:
            accuracy = dict()
            confusion_matrix = self.confusion_matrix
            for snr in self.snr_to_index:
                accuracy[f'{snr:02d}dB'] = self._accuracy(confusion_matrix[f'{snr:02d}dB'])
            # For All SNR
            accuracy['AllSNR'] = self._accuracy(confusion_matrix['AllSNR'])
            self._as = accuracy
            self._cache_as = True

        return self._as

    @property
    def F1(self) -> Dict[str, Dict[str, float]]:
        if not self._cache_fs:
            f1_score = dict()
            confusion_matrix = self.confusion_matrix
            for snr in self.snr_to_index:
                f1_score[f'{snr:02d}dB'] = self._f1_score(confusion_matrix[f'{snr:02d}dB'])
            f1_score['AllSNR'] = self._f1_score(confusion_matrix['AllSNR'])
            self._fs = f1_score
            self._cache_fs = True

        return self._fs

    @property
    def precision_recall(self) -> Dict[Union[str, Any], Dict[str, Dict[str, ndarray]]]:
        if not self._cache_pr:
            # The function precision_recall is based on the link:
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

            pr = dict()

            for snr in self.snr_to_index:
                # For each class in different snr
                pr[snr] = self._precision_recall(self.pps_along_snr[snr], self.tps_along_snr[snr])

            pr['micro'] = self._precision_recall(self.pps, self.tps)
            self._pr = pr
            self._cache_pr = True

        return self._pr

    @property
    def roc(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        if not self._cache_roc:
            roc = dict(ovr=dict(), ovo=dict())

            # ROC using the OvR
            for snr in self.snr_to_index:
                # For each class in different snr
                roc['ovr'][snr] = self._roc_ovr(self.pps_along_snr[snr], self.tps_along_snr[snr])
                roc['ovo'][snr] = self._roc_ovo(self.pps_along_snr[snr], self.gts_along_snr[snr])

            roc['ovr']['micro'] = self._roc_ovr(self.pps, self.tps)
            roc['ovo']['micro'] = self._roc_ovo(self.pps, self.gts)
            self._roc = roc
            self._cache_roc = True

        return self._roc


def get_classification_eval_with_snr(ps, gs, ss, cs, ms, feas=None, centers=None):
    performance_generator = ClassificationMetricsWithSNRForSingle(ps, gs, ss, cs, feas=feas, centers=centers)
    eval_results = dict()
    for metric in ms:
        res = getattr(performance_generator, metric)
        save_res = dict()
        for key in res:
            if 'All' in key:
                save_res.update({metric: res[key]})
            elif 'Image' in key:
                save_res.update({metric: res[key]})
        eval_results.update(save_res)

    return eval_results
