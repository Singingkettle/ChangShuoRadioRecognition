import argparse
import os
import pickle
from multiprocessing import Process, RawArray

import zmq

from wtisp.common.fileio import load as IOLoad
from wtisp.common.utils import Config
from wtisp.dataset.zmq import ZMQConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Server node to deliver cache data for any process')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--num', type=int, default=None, help='number of zmq server')
    args = parser.parse_args()
    return args


def load_cache_data(cache_data_dir, ann_file, data_type, filter_config=None):
    ann_prefix = ann_file.split('.')[0]
    if data_type == 'co':
        file_suffix = '-filter_size_{:<.3f}_stride_{:<.3f}'.format(filter_config[0], filter_config[1]) + '-co.pkl'
    else:
        file_suffix = '-{}.pkl'.format(data_type)
    cache_data_path = os.path.join(cache_data_dir, ann_prefix + file_suffix)
    data = pickle.load(open(cache_data_path, 'rb'))

    return data

def zmq_server(server_index):
    print("Start ZMQ Server %02d" % server_index)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    tcp_address = "tcp://*:{}".format(ZMQConfig['port'] + server_index)
    socket.bind(tcp_address)

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("ZMQ Server %02d receives request: %s" % (server_index, message))

        message = message.decode('utf-8')
        run_mode, data_type, idx = message.split('-')
        try:
            item = data[run_mode][data_type][int(idx)]
            socket.send(item)
        except Exception as e:
            print(e)


args = parse_args()
print('Command Line Args:', args)

# load cfg
cfg = Config.fromfile(args.config)

# Load cache data

data = dict()
## if use iq
cache_data_dir = os.path.join(cfg.data_root, 'zmq_cache_pkl_by_byte_compress')
data['train'] = dict()
if 'iq' in cfg.data['train']:
    if cfg.data['train']['iq']:
        data['train']['iq'] = load_cache_data(cache_data_dir, cfg.data['train']['ann_file'], 'iq')
if 'ap' in cfg.data['train']:
    if cfg.data['train']['ap']:
        data['train']['ap'] = load_cache_data(cache_data_dir, cfg.data['train']['ann_file'], 'ap')
if 'co' in cfg.data['train']:
    if cfg.data['train']['co']:
        ann_file_path = os.path.join(cfg.data_root, cfg.data['train']['ann_file'])
        annos = IOLoad(ann_file_path)
        filter_config = annos['filters'][cfg.data['train']['filter_config']]
        data['train']['co'] = load_cache_data(cache_data_dir, cfg.data['train']['ann_file'], 'co', filter_config)

data['val'] = dict()
if 'iq' in cfg.data['val']:
    if cfg.data['val']['iq']:
        data['val']['iq'] = load_cache_data(cache_data_dir, cfg.data['val']['ann_file'], 'iq')
if 'ap' in cfg.data['val']:
    if cfg.data['val']['ap']:
        data['val']['ap'] = load_cache_data(cache_data_dir, cfg.data['val']['ann_file'], 'ap')
if 'co' in cfg.data['val']:
    if cfg.data['val']['co']:
        ann_file_path = os.path.join(cfg.data_root, cfg.data['val']['ann_file'])
        annos = IOLoad(ann_file_path)
        filter_config = annos['filters'][cfg.data['val']['filter_config']]
        data['val']['co'] = load_cache_data(cache_data_dir, cfg.data['val']['ann_file'], 'co', filter_config)

print('Load Data Done!')

for server_index in range(args.num):
    Process(target=zmq_server, args=(server_index)).start()
