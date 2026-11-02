"""Camera switching must change pixels, while keeping outgoing FK actions identical."""
import json
import os
from pathlib import Path
import unittest

import h5py
import numpy as np
import torch



class CameraSwitchTest(unittest.TestCase):
    def test_actual_training_and_validation_camera0_pixels_and_unchanged_actions(self):
        repo = Path(__file__).resolve().parents[1]
        rows = []
        for task in ('bimanual_rope', 'single_grasp', 'bimanual_sweep'):
            config = json.loads((repo/'configs'/f'iws_{task}.json').read_text())
            config['root'] = str(Path(os.environ.get('IWS_DATA_ROOT', 'data/slotworld-iws-128')) / task)
            if not Path(config['root']).exists():
                self.skipTest('IWS integration data unavailable')
            from datasets.get_dataset import get_video_dataset
            self.assertEqual(config['camera'], 'camera_0_color')
            for mode, length in [('train', 21), ('valid', 45)]:
                camera0 = get_video_dataset(config['ds'], config['root'], seq_len=length,
                    mode=mode, image_size=128, camera=config['camera'])
                camera1 = get_video_dataset(config['ds'], config['root'], seq_len=length,
                    mode=mode, image_size=128)
                self.assertEqual(camera0.stats_path, camera1.stats_path)
                self.assertEqual(camera0.stats_path.parent.name, 'train')
                for ep in (0, len(camera0.files)-1):
                    for start in (0, camera0.lengths[ep]-length):
                        video0, action0, *_ = camera0.read_clip(ep, start)
                        video1, action1, *_ = camera1.read_clip(ep, start)
                        with h5py.File(camera0.files[ep], 'r') as h:
                            pixels = h['obs/images/camera_0_color'][start:start+length]
                        np.testing.assert_array_equal(
                            video0.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy(), pixels)
                        torch.testing.assert_close(action0, action1, rtol=0, atol=0)
                        self.assertFalse(torch.equal(video0, video1))
                        self.assertTrue(torch.isfinite(action0).all())
                rows.append({'task':task, 'mode':mode, 'camera':camera0.camera,
                             'episodes_checked_for_camera_shape':len(camera0.files),
                             'pixel_and_action_checks':4})
                camera0.close(); camera1.close()
        print(json.dumps({'camera_switch_checks':rows}))


if __name__ == '__main__':
    unittest.main()
