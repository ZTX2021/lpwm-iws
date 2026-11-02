import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_airhockey import make_data
from datasets.airhockey_h5 import AirHockeyH5Dataset
from modules.modules import ImagePatcher, DLPPrior, BgDecoder, DLPDecoder
from utils.airhockey_runtime import RunRecorder


class CropTest(unittest.TestCase):
    def test_crop_keeps_exact_pixels_and_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); make_data(root)
            data=AirHockeyH5Dataset(root,'valid',sample_length=3,image_size=[72,128])
            frames=np.zeros((3,128,128,3),dtype=np.uint8)
            frames[...,0]=np.arange(128)[None,:,None]
            frames[...,1]=np.arange(128)[None,None,:]
            frames[...,2]=np.arange(3)[:,None,None]
            actions=np.arange(12,dtype=np.float32).reshape(3,4)
            with patch.object(data.store,'read',return_value=(frames,actions)):
                video,action,*_=data[0]
            torch.testing.assert_close(video,torch.from_numpy(frames[:,28:100].copy()).permute(0,3,1,2).float()/255)
            torch.testing.assert_close(action,torch.from_numpy(actions))

    def test_patch_coordinates_and_rectangular_render(self):
        patcher=ImagePatcher(image_size=[72,128],patch_size=8)
        x=torch.arange(3*72*128).reshape(1,3,72,128).float()
        patches=patcher.img_to_patches(x)
        self.assertEqual(patches.shape,(1,3,144,8,8))
        torch.testing.assert_close(patcher.patches_to_img(patches),x)
        fake=SimpleNamespace(patcher=patcher,kp_range=(-1,1),image_size=[72,128])
        points=DLPPrior.get_global_kp(fake,torch.ones(1,144,1,2))
        torch.testing.assert_close(points[0,-1,0],torch.ones(2))
        canvas=SimpleNamespace(feature_map_size=[72,128])
        out=DLPDecoder.translate_patches(canvas,torch.zeros(1,1,2),torch.ones(1,1,1,16,16))
        self.assertEqual(out.shape,(1,1,1,72,128))
        bg=BgDecoder(image_size=[72,128],learned_bg_feature_dim=5,bg_ch_mult=(1,1,1,2,4),num_res_blocks=1)
        self.assertEqual(bg(torch.zeros(1,5)).shape,(1,3,72,128))

    def test_auxiliary_nan_is_recorded_without_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder=RunRecorder({},SimpleNamespace(is_main_process=True),tmp)
            with self.assertWarns(UserWarning):
                recorder.log({'train/loss':1.,'val/ssim':float('nan')},1)
            row=json.loads((Path(tmp)/'progress.json').read_text())
            self.assertIsNone(row['val/ssim'])
            self.assertEqual(row['train/loss'],1.)
