"""Read-only IWS episode adapter with outgoing, next-achieved-pose actions."""
from collections import OrderedDict
import os
from pathlib import Path
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

TASK_DIMS = {'pusht': 4, 'bimanual_rope': 8, 'single_grasp': 4, 'bimanual_sweep': 4}


class IWSH5Dataset(Dataset):
    def __init__(self, root, mode, sample_length=21, image_size=128, camera="camera_1_color"):
        if camera not in ("camera_0_color", "camera_1_color"):
            raise ValueError(f"Unsupported IWS camera: {camera}")
        self.camera = camera
        self.image_key = "obs/images/" + camera
        self.root = Path(root)
        self.task = self.root.name
        if self.task not in TASK_DIMS or image_size != 128:
            raise ValueError('Expected a supported IWS task directory and native 128x128 images')
        self.mode = 'val' if mode in ('val', 'valid', 'validation') else mode
        if self.mode not in ('train', 'val'):
            raise ValueError('IWS provides train and val splits')
        self.sample_length = int(sample_length)
        self.action_dim = TASK_DIMS[self.task]
        self.files = sorted((self.root / self.mode).glob('episode_*.hdf5'),
                            key=lambda p: int(p.stem.split('_')[-1]))
        if not self.files:
            raise FileNotFoundError(self.root / self.mode)
        self.lengths = []
        for path in self.files:
            with h5py.File(path, 'r') as h:
                n = h['action'].shape[0]
                if h[self.image_key].shape != (n, 128, 128, 3):
                    raise ValueError(f'Invalid image shape: {path}')
                if h['action'].shape != (n, self.action_dim) or n < self.sample_length:
                    raise ValueError(f'Invalid action shape or short episode: {path}')
            if not self.cache_path(path).is_file():
                raise FileNotFoundError(self.cache_path(path))
            self.lengths.append(n)
        # The validation split must use training statistics, never its own NPZ.
        stats_name = ('action_stats_fk_next.npz' if self.task == 'pusht'
                      else f'action_stats_{self.task}_fk_next.npz')
        self.stats_path = self.root / 'train' / stats_name
        with np.load(self.stats_path, allow_pickle=False) as stats:
            self.minimum = stats['min'].astype(np.float32)
            maximum = stats['max'].astype(np.float32)
        if self.minimum.shape != (self.action_dim,) or not np.isfinite(self.minimum).all() or not np.isfinite(maximum).all():
            raise ValueError('Invalid action normalization statistics')
        self.span = np.where(maximum-self.minimum < 1e-8, 1., maximum-self.minimum)
        self.clips_per_episode = 4 if self.mode == 'train' else 1
        self._pid, self._handles, self._actions = None, OrderedDict(), {}

    def cache_path(self, path):
        folder = path.parent / '_fk_action_cache'
        if self.task != 'pusht':
            folder /= self.task
        return folder / (path.name + '.npy')

    def __len__(self):
        return len(self.files) * self.clips_per_episode

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, segment = divmod(index, self.clips_per_episode)
        if self.mode == 'val':
            return episode, 0
        count = self.lengths[episode] - self.sample_length + 1
        lo = segment * count // self.clips_per_episode
        hi = max(lo, (segment + 1) * count // self.clips_per_episode - 1)
        return episode, random.randint(lo, hi)

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def __getstate__(self):
        return dict(self.__dict__, _pid=None, _handles=OrderedDict(), _actions={})

    def read_clip(self, episode, start):
        length = self.sample_length
        if not 0 <= episode < len(self.files) or start < 0 or start+length > self.lengths[episode]:
            raise IndexError('Clip must remain inside one episode')
        if self._pid != os.getpid():
            self.close()
            self._actions = {}
            self._pid = os.getpid()
        path = self.files[episode]
        if path not in self._handles:
            self._handles[path] = h5py.File(path, 'r')
            if len(self._handles) > 32:
                _, h = self._handles.popitem(last=False)
                h.close()
        self._handles.move_to_end(path)
        images = self._handles[path][self.image_key][start:start+length]
        if episode not in self._actions:
            fk = np.load(self.cache_path(path), allow_pickle=False)
            if fk.shape != (self.lengths[episode], self.action_dim) or not np.isfinite(fk).all():
                raise ValueError(f'Invalid FK cache: {path}')
            self._actions[episode] = fk
        # Shift relative to the episode, not relative to an already sliced clip.
        indices = np.minimum(np.arange(start, start+length)+1, self.lengths[episode]-1)
        actions = self._actions[episode][indices].astype(np.float32, copy=True)
        actions = (actions-self.minimum) / self.span * 2. - 1.
        video = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
        return video, torch.from_numpy(actions), '', torch.empty(0), torch.ones(length, dtype=torch.int64)

    def __getitem__(self, index):
        return self.read_clip(*self.locate(index))
