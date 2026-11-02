"""Read-only, worker-local access to the existing SA HDF5 shards."""
import json
import os
from pathlib import Path
import h5py
import numpy as np


class EpisodeStore:
    def __init__(self, root, split):
        self.root = Path(root)
        manifest = json.loads((self.root / "h5_manifest.json").read_text())
        if manifest.get("format") != "langtable_action_h5_v1" or not manifest.get("complete"):
            raise ValueError("Expected a complete langtable_action_h5_v1 manifest")
        if manifest["action_dim"] != 4 or manifest["image_size"] != [128, 128]:
            raise ValueError("Expected 128x128 RGB and four action coordinates")
        self.episodes = manifest["splits"][split]
        self._pid = None
        self._files = {}

    def group(self, episode):
        # Never share an HDF5 handle across forked DataLoader workers.
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        item = self.episodes[episode]
        shard = item["shard"]
        if shard not in self._files:
            self._files[shard] = h5py.File(self.root / shard, "r")
        return self._files[shard]["episodes"][item["key"]]

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}

    def __getstate__(self):
        return dict(self.__dict__, _files={}, _pid=None)

    def read(self, episode, start, length, incoming_actions=False):
        item = self.episodes[episode]
        if start < int(incoming_actions) or start + length > item["length"]:
            raise IndexError("Clip must stay within one episode and have aligned actions")
        g = self.group(episode)
        images = g["images"][start:start + length]
        action_start = start - int(incoming_actions)
        actions = g["actions"][action_start:action_start + length]
        if images.shape != (length, 128, 128, 3) or actions.shape != (length, 4):
            raise ValueError("HDF5 shape differs from manifest")
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite action")
        return images, actions.astype(np.float32, copy=False)
