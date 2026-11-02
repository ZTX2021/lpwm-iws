import json
import os
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.iws_h5 import IWSH5Dataset, TASK_DIMS


class IWSTest(unittest.TestCase):
    def test_alignment_clip_end_episode_end_train_stats_and_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task, dim in TASK_DIMS.items():
                root = Path(tmp)/task
                stats_name = 'action_stats_fk_next.npz' if task == 'pusht' else f'action_stats_{task}_fk_next.npz'
                for split in ('train', 'val'):
                    folder = root/split; folder.mkdir(parents=True)
                    np.savez(folder/stats_name, min=np.zeros(dim), max=np.full(dim, 10 if split=='train' else 100))
                    for ep in range(2):
                        file = folder/f'episode_{ep}.hdf5'
                        frames = np.broadcast_to(np.arange(8,dtype=np.uint8)[:,None,None,None], (8,128,128,3))
                        with h5py.File(file, 'w') as h:
                            h.create_dataset('obs/images/camera_1_color', data=frames)
                            h.create_dataset('action', data=np.zeros((8,dim),np.float32))
                        cache = folder/'_fk_action_cache'
                        if task != 'pusht': cache /= task
                        cache.mkdir(parents=True,exist_ok=True)
                        np.save(cache/(file.name+'.npy'),np.repeat((np.arange(8)+ep*20).astype(np.float32)[:,None],dim,axis=1))
                data = IWSH5Dataset(root,'valid',sample_length=3)
                for ep in range(2):
                    for start in (0,2,5):
                        v,a,*_=data.read_clip(ep,start)
                        np.testing.assert_allclose((v[:,0,0,0]*255).numpy(),np.arange(start,start+3),atol=1e-5)
                        expected = (np.minimum(np.arange(start,start+3)+1,7)+ep*20)/10*2-1
                        np.testing.assert_allclose(a[:,0].numpy(),expected,atol=1e-6)
                batch=next(iter(DataLoader(data,batch_size=2,num_workers=2)))
                self.assertEqual(tuple(batch[0].shape),(2,3,3,128,128))
                self.assertEqual(tuple(batch[1].shape),(2,3,dim))
                with self.assertRaises(IndexError):data.read_clip(0,6)
                data.close()

    def test_real_data(self):
        base=Path(os.environ.get('IWS_DATA_ROOT', 'data/slotworld-iws-128'))
        if not base.exists(): self.skipTest('IWS integration data unavailable')
        rows=[]
        for task,dim in TASK_DIMS.items():
            for split in ('train','val'):
                d=IWSH5Dataset(base/task,split)
                self.assertEqual(d.stats_path.parent.name,'train')
                for ep in (0,len(d.files)-1):
                    for start in (0,d.lengths[ep]-21):
                        video,action,*_=d.read_clip(ep,start)
                        self.assertEqual(tuple(video.shape),(21,3,128,128))
                        self.assertEqual(tuple(action.shape),(21,dim))
                        self.assertTrue(torch.isfinite(action).all())
                        self.assertTrue(0<=float(video.min())<=float(video.max())<=1)
                rows.append({'task':task,'split':split,'episodes':len(d.files),'clips':len(d),'action_dim':dim})
                d.close()
        print(json.dumps({'real_data_adapter_checks':rows}))


if __name__=='__main__': unittest.main()
