# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from csrr.registry import VISUALIZERS
from csrr.structures import DataSample
from .utils import get_adaptive_scale


@VISUALIZERS.register_module()
class UniversalVisualizer(Visualizer):
    """Universal Visualizer for multiple tasks.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.
    """
    DEFAULT_TEXT_CFG = {
        'family': 'monospace',
        'color': 'white',
        'bbox': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
        'verticalalignment': 'top',
        'horizontalalignment': 'left',
    }

    @master_only
    def visualize_cls(self,
                      image: np.ndarray,
                      data_sample: DataSample,
                      classes: Optional[Sequence[str]] = None,
                      draw_gt: bool = True,
                      draw_pred: bool = True,
                      draw_score: bool = True,
                      resize: Optional[int] = None,
                      rescale_factor: Optional[float] = None,
                      text_cfg: dict = dict(),
                      show: bool = False,
                      wait_time: float = 0,
                      out_file: Optional[str] = None,
                      name: str = '',
                      step: int = 0) -> None:
        """Visualize classification result.

        This method will draw an text box on the input image to visualize the
        information about classification, like the ground-truth label and
        prediction label.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            classes (Sequence[str], optional): The categories names.
                Defaults to None.
            draw_gt (bool): Whether to draw ground-truth labels.
                Defaults to True.
            draw_pred (bool): Whether to draw prediction labels.
                Defaults to True.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            resize (int, optional): Resize the short edge of the image to the
                specified length before visualization. Defaults to None.
            rescale_factor (float, optional): Rescale the image by the rescale
                factor before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts
                arguments of :meth:`mmengine.Visualizer.draw_texts`.
                Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        if self.dataset_meta is not None:
            classes = classes or self.dataset_meta.get('classes', None)

        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = mmcv.imresize(image, (resize, resize * h // w))
            else:
                image = mmcv.imresize(image, (resize * w // h, resize))
        elif rescale_factor is not None:
            image = mmcv.imrescale(image, rescale_factor)

        texts = []
        self.set_image(image)

        if draw_gt and 'gt_label' in data_sample:
            idx = data_sample.gt_label.tolist()
            class_labels = [''] * len(idx)
            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]
            labels = [str(idx[i]) + class_labels[i] for i in range(len(idx))]
            prefix = 'Ground truth: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        if draw_pred and 'pred_label' in data_sample:
            idx = data_sample.pred_label.tolist()
            score_labels = [''] * len(idx)
            class_labels = [''] * len(idx)
            if draw_score and 'pred_score' in data_sample:
                score_labels = [
                    f', {data_sample.pred_score[i].item():.2f}' for i in idx
                ]

            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]

            labels = [
                str(idx[i]) + score_labels[i] + class_labels[i]
                for i in range(len(idx))
            ]
            prefix = 'Prediction: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            'size': int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            '\n'.join(texts),
            **text_cfg,
        )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img
