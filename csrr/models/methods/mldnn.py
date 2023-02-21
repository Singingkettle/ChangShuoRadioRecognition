from .single_head_classifier import SingleHeadClassifier

from ..builder import TASKS


@TASKS.register_module()
class MLDNN(SingleHeadClassifier):
    def __init__(self, backbone, classifier_head, vis_fea=False):
        super(MLDNN, self).__init__(backbone, classifier_head, vis_fea, 'MLDNN')

    def simple_test(self, inputs, input_metas, **kwargs):
        x = self.extract_feat(**inputs)
        outs = self.classifier_head(x, self.vis_fea, True)

        results_list = []
        if isinstance(outs, dict):
            keys = list(outs.keys())
            batch_size = outs[keys[0]].shape[0]
            for idx in range(batch_size):
                item = dict()
                for key_str in keys:
                    if key_str != 'Final' or 'fea' in key_str:
                        save_name = self.method_name + '-' + key_str
                    else:
                        save_name = 'Final'
                    item[save_name] = outs2result(outs[key_str][idx, :])
                results_list.append(item)
        else:
            for idx in range(outs.shape[0]):
                result = outs2result(outs[idx, :])
                result = {'Final': result}
                results_list.append(result)
        return results_list