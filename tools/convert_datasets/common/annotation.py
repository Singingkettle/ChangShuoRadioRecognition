import copy


def init_annotations(mods, snrs, filter_config):
    annotations = {'filter_config': filter_config, 'modulations': mods, 'snrs': snrs, 'annotations': []}

    return annotations


def update_annotations(annotations, file_name, snr, modulation):
    annotations['annotations'].append(dict(file_name=file_name, snr=snr, modulation=modulation))

    return annotations


def combine_two_annotations(annotations1, annotations2):
    combine_annotation = copy.deepcopy(annotations1)
    update_list = ['annotations']
    for key_name in update_list:
        combine_annotation[key_name].extend(copy.deepcopy(annotations2[key_name]))

    return combine_annotation
