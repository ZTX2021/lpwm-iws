import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


def make_data(root):
    splits = {"train": [], "valid": []}
    for split, bases in [("train", [0, 20]), ("valid", [100])]:
        (root/split).mkdir()
        with h5py.File(root/split/"shard.h5", "w") as h:
            for i, base in enumerate(bases):
                key = f"{split}_{i}"
                g = h.create_group("episodes/"+key)
                frames = np.broadcast_to((np.arange(8)+base).astype(np.uint8)[:,None,None,None], (8,128,128,3)).copy()
                actions = np.repeat((np.arange(8)+base+1).astype(np.float32)[:,None],4,axis=1)
                g.create_dataset("images",data=frames)
                g.create_dataset("actions",data=actions)
                splits[split].append({"shard":f"{split}/shard.h5","key":key,"length":8})
    (root/"h5_manifest.json").write_text(json.dumps({"format":"langtable_action_h5_v1", "complete":True,"action_dim":4,"image_size":[128,128],"splits":splits}))

from unittest.mock import patch
from datasets.airhockey_h5 import AirHockeyH5Dataset


class AlignmentTest(unittest.TestCase):
    def test_transition_alignment_splits_and_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_data(root)
            data=AirHockeyH5Dataset(root,"train",sample_length=3)
            self.assertEqual(len(data),8)
            with patch("datasets.airhockey_h5.random.randint", side_effect=lambda lo,hi:lo):
                for i in range(len(data)):
                    video,action,*_=data[i]
                    frame=(video[:,0,0,0]*255).round()
                    torch.testing.assert_close(action[:,0], frame+1)
                    self.assertTrue(bool((frame[1:]-frame[:-1]==1).all()))
            valid=AirHockeyH5Dataset(root,"valid",sample_length=3)
            self.assertEqual(len(valid),1)
            self.assertEqual(round(float(valid[0][0][0,0,0,0])*255),100)
            # Parent has already opened a handle; children must reopen their own.
            batch=next(iter(DataLoader(data,batch_size=2,num_workers=2)))
            self.assertEqual(tuple(batch[0].shape),(2,3,3,128,128))
            with self.assertRaises(IndexError):data[len(data)]
            data.store.close();valid.store.close()


if __name__=="__main__":unittest.main()
