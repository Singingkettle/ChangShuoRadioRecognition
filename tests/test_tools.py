# Copyright (c) ShuoChang. All rights reserved.
import os.path
from pathlib import Path
from subprocess import PIPE, Popen
from unittest import TestCase

from mmengine.config import Config

MMPRE_ROOT = Path(__file__).parent.parent
ASSETS_ROOT = Path(__file__).parent / 'data'


class TestVerifyAMCDataset(TestCase):

    def setUp(self):
        self.dir = os.path.join(ASSETS_ROOT, 'dataset', 'amc')
        dataset_cfg = dict(
            type='AMCDataset',
            ann_file=os.path.join(self.dir, 'anno.json'),
            pipeline=[dict(type='LoadImageFromFile')],
            data_root=str(ASSETS_ROOT / 'dataset'),
        )
        ann_file = '\n'.join(['a/2.JPG 0', 'b/2.jpeg 1', 'b/subb/3.jpg 1'])
        (self.dir / 'ann.txt').write_text(ann_file)
        config = Config(dict(train_dataloader=dict(dataset=dataset_cfg)))
        self.config_file = Path(self.tmpdir.name) / 'config.py'
        config.dump(self.config_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/misc/verify_dataset.py',
            str(self.config_file),
            '--out-path',
            str(self.dir / 'log.log'),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn(
            f"{ASSETS_ROOT / 'dataset/a/2.JPG'} cannot be read correctly",
            out.decode().strip())
        self.assertTrue((self.dir / 'log.log').exists())
