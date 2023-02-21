import argparse
import os
import pickle
from itertools import chain, combinations

import numpy as np
from cvxopt import solvers, matrix, spdiag, log, div, mul
from scipy.special import softmax

from csrr.common.fileio import load as IOLoad


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition generate logits for knowledge distillation!')
    parser.add_argument('--work_dir', help='the dir to format logs and models')
    parser.add_argument('-config', '--configs', nargs='+')
    parser.add_argument('--anno_dir', help='the dir to format annotations')
    parser.add_argument('--anno_name', help='the file name of annotations', type=str, default='test.json',
                        choices={'train_and_val.json', 'train.json', 'val.json', 'test.json'})
    args = parser.parse_args()
    return args


def powerset(s):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def generate_targets(ann_file):
    annos = IOLoad(ann_file)
    anno_data = annos['data']
    mods_dict = annos['mods']
    targets = np.zeros((len(anno_data), len(mods_dict.keys())), dtype=np.float64)
    for item_index, item in enumerate(anno_data):
        targets[item_index, item['ann']['labels'][0]] = 1

    return targets.reshape(1, -1)


def generate_logits(pre_file):
    pre_data = np.load(pre_file)
    pre_data = pre_data.astype(np.float64)
    pre_data = softmax(pre_data, axis=1)

    return pre_data.reshape(1, -1)


def ridge_regression(p, t, l):
    n = p.shape[0]
    G = -1 * np.eye(n, dtype=np.float64)
    h = np.zeros((n, 1), dtype=np.float64)
    P = np.dot(p, p.transpose()) + l / 2
    q = np.dot(p, t.transpose())
    A = np.ones((1, n), dtype=np.float64)
    b = np.ones((1, 1), dtype=np.float64)

    G = matrix(G)
    h = matrix(h)
    P = matrix(P)
    q = matrix(q)
    A = matrix(A)
    b = matrix(b)

    sol = solvers.qp(P, q, G, h, A=A, b=b)

    return np.array(sol['x']), float(sol['primal objective'])


def cross_entropy(p, t, l):
    n = p.shape[0]
    G = -1 * np.eye(n, dtype=np.float64)
    h = np.zeros((n, 1), dtype=np.float64)
    A = np.ones((1, n), dtype=np.float64)
    b = np.ones((1, 1), dtype=np.float64)
    l = float(l)

    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    p = matrix(p)
    t = matrix(t)

    h_d = np.ones((n, 1), dtype=np.float64)
    h_d = matrix(h_d)

    # def F(x=None, z=None):
    #     if x is None:
    #         return 0, matrix(0.0, (n, 1))
    #     x = np.array(x)
    #     y = np.dot(p.transpose(), x)
    #     f = -np.dot(t, np.log(y))/88000 + 0.5*l*np.dot(x.transpose(), x)
    #     f = float(f)
    #     y = y.transpose()
    #     d1 = t/(y+ np.finfo(np.float64).eps)
    #     Df = -np.sum(d1*p, axis=1)/88000+ l*x
    #     Df = matrix(Df.transpose())
    #     if z is None:
    #         return f, Df
    #     d2 = p/(y+ np.finfo(np.float64).eps)
    #     H = (np.dot(d1*d2, p.transpose())/88000 + l)
    #
    #     H = matrix(H)
    #     return f, Df, H

    def F(x=None, z=None):
        if x is None:
            x0 = np.zeros((n, 1), dtype=np.float64)
            x0[0, 0] = 1
            return 0, matrix(x0)

        y = p.T * x
        f1 = -t * log(y) / 88000
        f2 = x.T * x * 0.5 * l
        f = sum(f1 + f2)
        d1 = div(t.T, y)
        Df = -p * d1 / 88000 + l * x
        if z is None:
            return f, Df.T
        d1 = div(d1, y)
        H = (mul(h_d * d1.T, p) * p.T / 88000 + l) * z[0]

        return f, Df.T, H

    # w1 = np.ones((2, 1), dtype=np.float64)/2
    # w2 = np.ones((2, 1), dtype=np.float64)/2
    # w3 = np.ones((2, 1), dtype=np.float64)/2
    # w2[0, 0] -= 0.0000000001
    # w3[0, 0] += 0.0000000001
    #
    # f1, Df1, H = F(x=matrix(w1), z=[0])
    # f2, Df2 = F(x=matrix(w2))
    # f3, Df3 = F(x=matrix(w3))
    # print(Df1[0, 0] - (f3-f2)/0.0000000002)
    # print(H[0, 0] - (Df3[0, 0]-Df2[0, 0])/0.0000000002)

    sol = solvers.cp(F, G=G, h=h, A=A, b=b)

    return np.array(sol['x']), float(sol['primal objective'])


def acent(A, b):
    m, n = A.size

    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n, 1))
        if min(x) <= 0.0: return None
        f = -sum(log(x))
        Df = -(x ** -1).T
        if z is None: return f, Df
        H = spdiag(z[0] * x ** -2)
        return f, Df, H

    return solvers.cp(F, A=A, b=b)['x']


if __name__ == '__main__':
    args = parse_args()

    # A = np.random.rand(4, 4)
    # b = np.random.rand(4, 1)
    # acent(matrix(A), matrix(b))
    configs = args.configs
    work_dir = args.work_dir
    anno_dir = args.anno_dir
    anno_name = args.anno_name

    targets = generate_targets(os.path.join(anno_dir, anno_name))

    logits_dict = dict()
    for config in configs:
        if os.path.isfile(os.path.join(work_dir, config, 'format_out/pre.npy')):
            pre_file_path = os.path.join(work_dir, config, 'format_out/pre.npy')
        elif os.path.isfile(os.path.join(work_dir, config, 'format_out/merge_pre.npy')):
            pre_file_path = os.path.join(work_dir, config, 'format_out/merge_pre.npy')
        else:
            raise ValueError('Please generate the prediction results: {}'.format(config))
        logits_dict[config] = generate_logits(pre_file_path)

    configs_sets = list(powerset(configs))

    r_dict = dict(w=[], o=[], c=[], r=[])
    c_dict = dict(w=[], o=[], c=[], r=[])
    lambda_dict = dict()
    for lambda_index, lambda_val in enumerate(np.arange(0.01, 1, 0.1)):
        r_min_val = 1000000000
        r_o = None
        r_subset_index = None
        r_subset_w = None

        c_min_val = 1000000000
        c_o = None
        c_subset_index = None
        c_subset_w = None

        for subset_index, subset in enumerate(configs_sets):
            if 'mldnn_mlnetv5_640_0.0004_0.5_deepsig_201610A' in subset:
                if len(subset) > 0:
                    print('lambda: {:.2f}-configs-{}'.format(lambda_val, '+'.join(subset)))
                    pre_matrix = []
                    for config in subset:
                        pre_matrix.append(logits_dict[config])
                    pre_matrix = np.vstack(pre_matrix)
                    r_w, r_o = ridge_regression(pre_matrix, targets, lambda_val)
                    c_w, c_o = cross_entropy(pre_matrix, targets, lambda_val)
                    if r_o < r_min_val:
                        r_min_val = r_o
                        r_subset_index = subset_index
                        r_subset_w = r_w
                    if c_o < c_min_val:
                        c_min_val = c_o
                        c_subset_index = subset_index
                        c_subset_w = c_w

        r_dict['w'].append(r_subset_w)
        r_dict['o'].append(r_o)
        r_dict['c'].append(configs_sets[r_subset_index])
        r_dict['r'].append(lambda_val)

        c_dict['w'].append(c_subset_w)
        c_dict['o'].append(c_o)
        c_dict['c'].append(configs_sets[c_subset_index])
        c_dict['r'].append(lambda_val)

        lambda_dict[lambda_val] = lambda_index
        break

    r_dict['lambda'] = lambda_dict
    r_dict['work_dir'] = args.work_dir

    c_dict['lambda'] = lambda_dict
    c_dict['work_dir'] = args.work_dir

    if os.path.isfile(os.path.join(args.anno_dir, 'rg_metadata_multi-teacher_in_knowledge_distillation.pkl')):
        os.remove(os.path.join(args.anno_dir, 'rg_metadata_multi-teacher_in_knowledge_distillation.pkl'))
    if os.path.isfile(os.path.join(args.anno_dir, 'ce_metadata_multi-teacher_in_knowledge_distillation.pkl')):
        os.remove(os.path.join(args.anno_dir, 'ce_metadata_multi-teacher_in_knowledge_distillation.pkl'))

    pickle.dump(r_dict,
                open(os.path.join(args.anno_dir, 'rg_metadata_multi-teacher_in_knowledge_distillation.pkl'), 'wb'))
    pickle.dump(c_dict,
                open(os.path.join(args.anno_dir, 'ce_metadata_multi-teacher_in_knowledge_distillation.pkl'), 'wb'))
