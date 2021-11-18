import cv2
from pathlib import Path
import os
import torch
import numpy as np
from glob import glob
import data_util
import util
from collections import defaultdict
import load_llff
from PIL import Image
import skimage.filters


class SceneInstanceDataset(torch.utils.data.Dataset):
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,
                 img_sidelength=None,
                 num_images=None,
                 cache=None):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = Path(instance_dir)
        self.cache = cache

        color_dir, pose_dir, depth_dir, param_dir = [self.instance_dir / s for s in ['rgb', 'pose', 'depth_exr', 'params']]
        self.has_depth = depth_dir.exists()
        self.has_params = param_dir.exists()

        if not color_dir.exists():
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.pose_paths = sorted(pose_dir.glob("*.txt"))
        self.param_paths = sorted(param_dir.glob("*.txt"))
        self.depth_paths = sorted(depth_dir.glob("*.exr"))
        self.instance_name = os.path.basename(self.instance_dir)

        if specific_observation_idcs is not None:
            self.color_paths = util.pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = util.pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = util.pick(self.param_paths, specific_observation_idcs)
            self.depth_paths = util.pick(self.depth_paths, specific_observation_idcs)
        elif num_images is not None:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = util.pick(self.color_paths, idcs)
            self.pose_paths = util.pick(self.pose_paths, idcs)
            self.param_paths = util.pick(self.param_paths, idcs)
            self.depth_paths = util.pick(self.depth_paths, idcs)

        dummy_img = data_util.load_rgb(self.color_paths[0])
        self.org_sidelength = dummy_img.shape[1]

        if self.org_sidelength < self.img_sidelength:
            uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32).transpose(1, 2, 0)
            self.intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                             trgt_sidelength=self.img_sidelength)
        else:
            uv = np.mgrid[0:self.org_sidelength, 0:self.org_sidelength].astype(np.int32).transpose(1, 2, 0)
            uv = cv2.resize(uv, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
            self.intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                             trgt_sidelength=self.org_sidelength)

        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        self.uv = uv.reshape(-1, 2).float()

        self.intrinsics = torch.Tensor(self.intrinsics).float()

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return min(len(self.pose_paths), len(self.color_paths))

    def __getitem__(self, idx):
        key = f'{self.instance_idx}_{idx}'
        if (self.cache is not None) and (key in self.cache):
            rgb, pose, edges, *depth = self.cache[key]
        else:
            rgb = data_util.load_rgb(self.color_paths[idx])
            edges = skimage.filters.sobel(rgb)
            pose = data_util.load_pose(self.pose_paths[idx])
            to_cache = [rgb, pose, edges]

            if self.has_depth:
                depth = data_util.load_depth(str(self.depth_paths[idx]))
                to_cache.append(depth)

            if (self.cache is not None) and (key not in self.cache):
                self.cache[key] = to_cache

        rgb = cv2.resize(rgb, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
        rgb = rgb.reshape(-1, 3)

        edges = cv2.resize(edges, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
        edges = edges.reshape(-1, 3)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
            "rgb": torch.from_numpy(rgb).float(),
            "cam2world": torch.from_numpy(pose).float(),
            "uv": self.uv,
            "edges":torch.from_numpy(edges).float(),
            "intrinsics": self.intrinsics,
            "height_width": torch.from_numpy(np.array([self.img_sidelength, self.img_sidelength])),
            "instance_name": self.instance_name
        }

        if self.has_depth:
            depth = cv2.resize(depth, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
            depth = depth.reshape(-1, 1)
            sample["depth"] = torch.from_numpy(depth).float()

        return sample


def get_instance_datasets(root, max_num_instances=None, specific_observation_idcs=None,
                          cache=None, sidelen=None, max_observations_per_instance=None):
    instance_dirs = sorted(glob(os.path.join(root, "*/")))
    assert (len(instance_dirs) != 0), f"No objects in the directory {root}"

    if max_num_instances != None:
        instance_dirs = instance_dirs[:max_num_instances]

    all_instances = [SceneInstanceDataset(instance_idx=idx, instance_dir=dir,
                                          specific_observation_idcs=specific_observation_idcs, img_sidelength=sidelen,
                                          cache=cache, num_images=max_observations_per_instance)
                     for idx, dir in enumerate(instance_dirs)]
    return all_instances


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 num_context, num_trgt, root_dir,
                 vary_context_number=False,
                 query_sparsity=None,
                 img_sidelength=None,
                 max_num_instances=None,
                 max_observations_per_instance=None,
                 specific_observation_idcs=None,
                 test=False,
                 test_context_idcs=None,
                 cache=None):

        self.num_context = num_context
        self.num_trgt = num_trgt
        self.query_sparsity = query_sparsity
        self.img_sidelength = img_sidelength
        self.vary_context_number = vary_context_number
        self.cache = cache
        self.test = test
        self.test_context_idcs = test_context_idcs

        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))
        print(f"Root dir {root_dir}, {len(self.instance_dirs)} instances")

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances is not None:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance,
                                                   cache=cache)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)
        
    def sparsify(self, dict, sparsity):
        new_dict = {}
        if sparsity is None:
            return dict
        else:
            # Sample upper_limit pixel idcs at random.
            rand_idcs = np.random.choice(self.img_sidelength**2, size=sparsity, replace=False)
            for key in ['rgb', 'uv']:
                new_dict[key] = dict[key][rand_idcs]

            for key, v in dict.items():
                if key not in ['rgb', 'uv']:
                    new_dict[key] = dict[key]

            return new_dict

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with which images are loaded."""
        self.img_sidelength = new_img_sidelength
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        if self.test:
            obj_idx = 0
            while idx >= 0:
                idx -= self.num_per_instance_observations[obj_idx]
                obj_idx += 1
            return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])
        else:
            return np.random.randint(self.num_instances), 0

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys:
                result[key].append(entry[key])

        for key in keys:
            try:
                result[key] = torch.stack(result[key], dim=0)
            except:
                continue

        return result

    def __getitem__(self, idx):
        context = []
        trgt = []
        post_input = []

        obj_idx, det_idx = self.get_instance_idx(idx)

        if self.vary_context_number:
            num_context = np.random.randint(1, self.num_context+1)

        if not self.test:
            try:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=False,
                                               size=self.num_context+self.num_trgt)
            except:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=True,
                                               size=self.num_context+self.num_trgt)

        for i in range(self.num_context):
            if self.test:
                sample = self.all_instances[obj_idx][self.test_context_idcs[i]]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i]]
            context.append(sample)

            if self.vary_context_number:
                if i < num_context:
                    context[-1]['mask'] = torch.Tensor([1.])
                else:
                    context[-1]['mask'] = torch.Tensor([0.])
            else:
                context[-1]['mask'] = torch.Tensor([1.])

        for i in range(self.num_trgt):
            if self.test:
                sample = self.all_instances[obj_idx][det_idx]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i+self.num_context]]

            post_input.append(sample)
            post_input[-1]['mask'] = torch.Tensor([1.])

            sub_sample = self.sparsify(sample, self.query_sparsity)
            trgt.append(sub_sample)

        # trgt.append(context[0])

        post_input = self.collate_fn(post_input)
        context = self.collate_fn(context)
        trgt = self.collate_fn(trgt)

        return {'context': context, 'query': trgt, 'post_input': post_input}, trgt


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
