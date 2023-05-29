import copy
import os.path
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import List
from typing import Sequence
import pickle
from tqdm import tqdm
import numpy as np
from mmdet.datasets.api_wrappers import COCO
from pycocotools.cocoeval import COCOeval as _COCOeval
import json

from .builder import DATASETS
from .custom import CustomDataset
from ..common import dump, load


class COCOeval(_COCOeval):
    def __int__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)

    def summarize(self):
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=9):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


@DATASETS.register_module()
class CSRRDataset(CustomDataset):
    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False,
                 preprocess=None, evaluate=None, format=None, CLASSES=None, is_only_det=True):
        super(CSRRDataset, self).__init__(ann_file, pipeline, data_root, test_mode, preprocess, evaluate, format)

        if is_only_det:
            self.CLASSES = ['signal']
        else:
            self.CLASSES = self.data_infos['modulations']

        self.is_only_det = is_only_det

        coco_style_dataset = dict(info='A CoCo style signal detection data', licenses='None',
                                  images=[], annotations=[], categories=[])

        outfile_path = osp.join(self.data_root, f'coco-{ann_file}')
        sr = self.data_infos['sample_rate']
        sn = self.data_infos['sample_num']

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
                if is_only_det:
                    anno['category_id'] = 0
                else:
                    anno['category_id'] = self.CLASSES.index(item['modulation'][anno_id])
                x = (item['center_frequency'][anno_id] / sr + 0.5) * sn
                w = item['bandwidth'][anno_id] / sr * sn
                anno['bbox'] = [x - 0.5 * w, 0, w, 1]
                anno['area'] = 1 * w
                anno['segmentation'] = [[x - 0.5 * w, 0, x - 0.5 * w, 0 + 1, x + 0.5 * w, 0 + 1, x + 0.5 * w, 0]]
                anno['snr'] = item['snr'][anno_id]
                item_annotations.append(anno)
                global_anno_id += 1
            coco_style_dataset['annotations'].extend(item_annotations)
        for class_name in self.CLASSES:
            coco_style_dataset['categories'].append(
                dict(supercategory='signal', id=self.CLASSES.index(class_name), name=class_name))
        dump(coco_style_dataset, outfile_path)

        self._coco_api = COCO(outfile_path)
        self.proposal_nums = [5, 6, 7]
        self.iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    def get_ann_info(self, idx):
        results = dict()
        results['snr'] = self.data_infos['annotations'][idx]['snr']
        results['file_name'] = self.data_infos['annotations'][idx]['file_name']
        results['image_id'] = idx

        results['center_frequency'] = self.data_infos['annotations'][idx]['center_frequency']
        results['bandwidth'] = self.data_infos['annotations'][idx]['bandwidth']
        results['sample_rate'] = self.data_infos['sample_rate']
        results['sample_num'] = self.data_infos['sample_num']
        if not self.is_only_det:
            results['label'] = [self.CLASSES.index(mod) for mod in self.data_infos['annotations'][idx]['modulation']]
        else:
            results['label'] = [1] * len(results['center_frequency'])

        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['data_folder'] = 'sequence_data/iq'
        results['inputs'] = dict()
        return results

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

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
        coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs
        coco_eval.params.areaRng = [[0, 130], [0, 90], [90, 110], [110, 130]]

        # mapping of cocoEval.stats

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

        metric_items = coco_metric_names.keys()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = coco_eval.stats[coco_metric_names[metric_item]]
            eval_results[key] = float(f'{round(val, 4)}')

        ap = coco_eval.stats[:6]
        logger.info(f'{metric}_mAP_copypaste: {ap[0]:.4f} '
                    f'{ap[1]:.4f} {ap[2]:.4f} {ap[3]:.4f} '
                    f'{ap[4]:.4f} {ap[5]:.4f}')

        return eval_results


    def two_stage(self, results, amc_iou=0.5, mode=None):
        ann_file = os.path.basename(self.ann_file)
        self.cache_data = pickle.load(open(osp.join(self.data_root, 'cache', ann_file.replace('.json', '_iq.pkl')), 'rb'))
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
        coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)
        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs
        coco_eval.params.areaRng = [[0, 120], [120, ], [105, 140]]
        coco_eval.evaluate()

        annotations = {'modulations': self.data_infos['modulations'], 'snrs': self.data_infos['snrs'], 'annotations': []}
        global_iq_id = 0
        for i in tqdm(range(len(self))):
            ann_info = self.get_ann_info(i)
            idx = self.cache_data['lookup_table'][ann_info['file_name']]
            iq = self.cache_data['data'][idx]
            iq = np.squeeze(iq)
            iq = iq[0, :] + 1j * iq[1, :]
            ft = np.fft.fft(iq)
            ft = np.fft.fftshift(ft)
            for bi in range(results[i]['bboxes'].shape[0]):
                x1 = results[i]['bboxes'][bi, 0]
                x2 = results[i]['bboxes'][bi, 2]
                x1 = int(max(np.floor(x1), 0))
                x2 = int(min(np.round(x2), 1199))
                aft = copy.deepcopy(ft)
                aft[:x1] = 0
                aft[x2:] = 0
                iq_ft = np.fft.ifftshift(aft)
                iq = np.fft.ifft(iq_ft)
                i_seq = np.real(iq)
                q_seq = np.imag(iq)
                iq = np.vstack((i_seq, q_seq))
                max_iou = np.max(coco_eval.ious[(i, 0)][bi, :])
                max_iou_index = np.argmax(coco_eval.ious[(i, 0)][bi, :])
                iq_file_name = 'ts-{}-{:0>12d}.npy'.format(mode, global_iq_id)
                iq_path = osp.join(self.data_root, 'sequence_data/iq', iq_file_name)
                np.save(iq_path, iq)
                ann = dict()
                ann['file_name'] = iq_file_name
                if max_iou >= amc_iou:
                    ann['modulation'] = self.data_infos['annotations'][idx]['modulation'][max_iou_index]
                    ann['snr'] = self.data_infos['annotations'][idx]['snr'][max_iou_index]
                else:
                    ann['modulation'] = 'none'
                    ann['snr'] = min(self.data_infos['snrs'])
                annotations['annotations'].append(ann)
                global_iq_id += 1
        annotations['modulations'].append('none')
        json.dump(annotations, open(f'{self.data_root}/ts-{ann_file}', 'w'), indent=4, sort_keys=True)





