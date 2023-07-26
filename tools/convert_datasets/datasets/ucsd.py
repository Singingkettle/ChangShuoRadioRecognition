import os.path as osp

from .deepsig import DeepSigBase


class UCSDRML22(DeepSigBase):
    def __init__(self, root_dir, version, data_ratios):
        super(UCSDRML22, self).__init__(root_dir, version, data_ratios)
        self.data_name = 'RML22.pickle.01A'
        self.mod2mod = {"8PSK": '8PSK', "AM-DSB": "AM-DSB", "BPSK": "BPSK", "CPFSK": "CPFSK",
                        "GFSK": "GFSK", "PAM4": "4PAM", "QAM16": "16QAM", "QAM64": "64QAM", "QPSK": "QPSK",
                        "WBFM": "WBFM"}
        self.organization = 'UCSD'
        self.data_dir = osp.join(self.root_dir, self.organization, self.version)
