import os
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from typing import List
from terminaltables import AsciiTable
from typing import Sequence
import itertools
import numpy as np
import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval
from mmdet.datasets.api_wrappers import COCO

from .builder import DATASETS
from .custom import CustomDataset
from ..common import dump, load


@DATASETS.register_module()
class CSRRDataset(CustomDataset):
    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False,
                 preprocess=None, evaluate=None, format=None, CLASSES=None):
        super(CSRRDataset, self).__init__(ann_file, pipeline, data_root, test_mode, preprocess, evaluate, format)

        self.CLASSES = self.data_infos['modulations']
        self.is_det = False

        coco_style_dataset = dict(info='A CoCo style signal detection data', licenses='None',
                                  images=[], annotations=[], categories=[])

        outfile_path = osp.join(self.data_root, f'coco-{ann_file}')
        if not os.path.isfile(outfile_path):
            global_anno_id = 0
            for id, item in enumerate(self.data_infos['annotations']):
                signal = dict(license=4, file_name=item['file_name'], coco_url='',
                              height=3, width=128, data_captured='', flickr_url='', id=id)
                coco_style_dataset['images'].append(signal)
                item_annotations = []
                for anno_id in range(len(item['modulation'])):
                    anno = dict()
                    anno['iscrowd'] = 0
                    anno['image_id'] = id
                    anno['id'] = global_anno_id
                    if CLASSES is None:
                        anno['category_id'] = 'signal'
                    else:
                        anno['category_id'] = self.CLASSES.index(item['modulation'][anno_id])
                    x = item['center_frequency'][anno_id] + 0.5 * item['bandwidth'][anno_id] + \
                        0.5 * self.data_infos['sample_rate']
                    w = item['bandwidth'][anno_id]
                    x = x / self.data_infos['sample_rate'] * self.data_infos['sample_num']
                    w = w / self.data_infos['sample_rate'] * self.data_infos['sample_num']
                    anno['bbox'] = [x, 0, w, 1]
                    anno['area'] = 1 * w
                    anno['segmentation'] = [[x, 0, x, 0+1, x + w, 0+1, x + w, 0]]
                    anno['snr'] = item['snr'][anno_id]
                    item_annotations.append(anno)
                    global_anno_id += 1
                coco_style_dataset['annotations'].extend(item_annotations)
            for mod in self.data_infos['modulations']:
                coco_style_dataset['categories'].append(
                    dict(supercategory='signal', id=self.CLASSES.index(mod), name=mod))
            dump(coco_style_dataset, outfile_path)

        self._coco_api = COCO(outfile_path)

    def get_ann_info(self, idx):
        results = dict()
        results['snr'] = self.data_infos['annotations'][idx]['snr']
        results['file_name'] = self.data_infos['annotations'][idx]['file_name']
        results['modulation'] = [self.CLASSES.index(mod) for mod in self.data_infos['annotations'][idx]['modulation']]
        results['center_frequency'] = self.data_infos['annotations'][idx]['center_frequency']
        results['bandwidth'] = self.data_infos['annotations'][idx]['bandwidth']
        results['sample_rate'] = self.data_infos['sample_rate']
        results['sample_num'] = self.data_infos['sample_num']
        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['data_folder'] = 'sequence_data/iq'
        results['inputs'] = dict()
        return results

    def evaluate(self, results, logger=None):
        tmp_dir = tempfile.TemporaryDirectory()
        outfile_prefix = osp.join(tmp_dir.name, 'preds')
        self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.CLASSES)
        self.img_ids = self._coco_api.get_img_ids()
        result_files = self.results2json(results, outfile_prefix)
        eval_results = OrderedDict()
        iou_type = 'bbox'
        metric = 'bbox'
        predictions = load(result_files['bbox'])
        coco_dt = self._coco_api.loadRes(predictions)
        coco_eval = _COCOeval(self._coco_api, coco_dt, iou_type)

        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs

        # mapping of cocoEval.stats
        metric_items = ['mAP', 'mAP_50', 'mAP_75']
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = coco_eval.stats[coco_metric_names[metric_item]]
            eval_results[key] = float(f'{round(val, 3)}')

        ap = coco_eval.stats[:6]
        logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                    f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
