import cv2
import os
import torch
import numpy as np
from glob import glob
import data_util
import util
from collections import defaultdict
import h5py


class SceneInstanceDataset(torch.utils.data.Dataset):
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=None,
                 cache=None):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir
        self.cache = cache
        self.has_depth = False

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))
        self.instance_name = os.path.basename(os.path.dirname(self.instance_dir))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = util.pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = util.pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = util.pick(self.param_paths, specific_observation_idcs)
        elif num_images is not None:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = util.pick(self.color_paths, idcs)
            self.pose_paths = util.pick(self.pose_paths, idcs)
            self.param_paths = util.pick(self.param_paths, idcs)

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

    def __len__(self):
        return min(len(self.pose_paths), len(self.color_paths))

    def __getitem__(self, idx):
        key = f'{self.instance_idx}_{idx}'
        if (self.cache is not None) and (key in self.cache):
            rgb, pose = self.cache[key]
        else:
            rgb = data_util.load_rgb(self.color_paths[idx])
            pose = data_util.load_pose(self.pose_paths[idx])

            if (self.cache is not None) and (key not in self.cache):
                self.cache[key] = rgb, pose

        rgb = cv2.resize(rgb, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
        rgb = rgb.reshape(-1, 3)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
            "rgb": torch.from_numpy(rgb).float(),
            "cam2world": torch.from_numpy(pose).float(),
            "uv": self.uv,
            "intrinsics": self.intrinsics,
            "height_width": torch.from_numpy(np.array([self.img_sidelength, self.img_sidelength])),
            "instance_name": self.instance_name
        }

        return sample


class SceneInstanceDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, instance_idx, instance_ds, img_sidelength, instance_name, specific_observation_idcs=None,
                 num_images=None, cache=None):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.cache = cache
        self.instance_ds = instance_ds
        self.has_depth = False

        self.color_keys = sorted(list(instance_ds['rgb'].keys()))
        self.pose_keys = sorted(list(instance_ds['pose'].keys()))
        self.instance_name = instance_name

        if specific_observation_idcs is not None:
            self.color_keys = util.pick(self.color_keys, specific_observation_idcs)
            self.pose_keys = util.pick(self.pose_keys, specific_observation_idcs)
        elif num_images is not None:
            idcs = np.linspace(0, stop=len(self.color_keys), num=num_images, endpoint=False, dtype=int)
            self.color_keys = util.pick(self.color_keys, idcs)
            self.pose_keys = util.pick(self.pose_keys, idcs)

        dummy_img = data_util.load_rgb_hdf5(self.instance_ds, self.color_keys[0])
        self.org_sidelength = dummy_img.shape[1]

        if self.org_sidelength < self.img_sidelength:
            uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32).transpose(1, 2, 0)
            self.intrinsics, _, _ = util.parse_intrinsics_hdf5(instance_ds['intrinsics.txt'],
                                                                  trgt_sidelength=self.img_sidelength)
        else:
            uv = np.mgrid[0:self.org_sidelength, 0:self.org_sidelength].astype(np.int32).transpose(1, 2, 0)
            uv = cv2.resize(uv, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
            self.intrinsics, _, _ = util.parse_intrinsics_hdf5(instance_ds['intrinsics.txt'],
                                                                  trgt_sidelength=self.org_sidelength)

        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        self.uv = uv.reshape(-1, 2).float()
        self.intrinsics = torch.from_numpy(self.intrinsics).float()

    def __len__(self):
        return min(len(self.pose_keys), len(self.color_keys))

    def __getitem__(self, idx):
        key = f'{self.instance_idx}_{idx}'
        if (self.cache is not None) and (key in self.cache):
            rgb, pose = self.cache[key]
        else:
            rgb = data_util.load_rgb_hdf5(self.instance_ds, self.color_keys[idx])
            pose = data_util.load_pose_hdf5(self.instance_ds, self.pose_keys[idx])

            if (self.cache is not None) and (key not in self.cache):
                self.cache[key] = rgb, pose

        rgb = cv2.resize(rgb, (self.img_sidelength, self.img_sidelength), interpolation=cv2.INTER_NEAREST)
        rgb = rgb.reshape(-1, 3)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
            "rgb": torch.from_numpy(rgb).float(),
            "cam2world": torch.from_numpy(pose).float(),
            "uv": self.uv,
            "intrinsics": self.intrinsics,
            "instance_name": self.instance_name
        }
        return sample


def get_num_instances(data_root):
    if 'hdf5' in data_root:
        file = h5py.File(data_root, 'r')
        instances = list(file.keys())
    else:
        instances = len(glob(os.path.join(data_root, "*/")))
    return len(instances)


def get_instance_datasets(root, max_num_instances=None, specific_observation_idcs=None,
                          cache=None, sidelen=None, max_observations_per_instance=None, start_idx=0):
    instance_dirs = sorted(glob(os.path.join(root, "*/")))
    assert (len(instance_dirs) != 0), f"No objects in the directory {root}"

    if max_num_instances != None:
        instance_dirs = instance_dirs[:max_num_instances]

    all_instances = [SceneInstanceDataset(instance_idx=idx+start_idx, instance_dir=dir,
                                          specific_observation_idcs=specific_observation_idcs, img_sidelength=sidelen,
                                          cache=cache, num_images=max_observations_per_instance)
                     for idx, dir in enumerate(instance_dirs)]
    return all_instances


def get_instance_datasets_hdf5(root, max_num_instances=None, specific_observation_idcs=None,
                               cache=None, sidelen=None, max_observations_per_instance=None,
                               start_idx=0):
    file = h5py.File(root, 'r')
    instances = sorted(list(file.keys()))
    print(f"File {root}, {len(instances)} instances")

    if max_num_instances is not None:
        instances = instances[:max_num_instances]

    all_instances = [SceneInstanceDatasetHDF5(instance_idx=idx+start_idx, instance_ds=file[instance_name],
                                              specific_observation_idcs=specific_observation_idcs,
                                              img_sidelength=sidelen, num_images=max_observations_per_instance,
                                              cache=cache, instance_name=instance_name)
                          for idx, instance_name in enumerate(instances)]
    return all_instances


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self, num_context, num_trgt, data_root,
                 vary_context_number=False, query_sparsity=None,
                 img_sidelength=None, max_num_instances=None,
                 max_observations_per_instance=None, specific_observation_idcs=None,
                 test=False, test_context_idcs=None, cache=None, start_idx=0):

        self.num_context = num_context
        self.num_trgt = num_trgt
        self.query_sparsity = query_sparsity
        self.img_sidelength = img_sidelength
        self.vary_context_number = vary_context_number
        self.cache = cache
        self.test = test
        self.test_context_idcs = test_context_idcs

        if 'hdf5' in data_root:
            self.all_instances = get_instance_datasets_hdf5(data_root, max_num_instances=max_num_instances,
                                                            specific_observation_idcs=specific_observation_idcs,
                                                            cache=cache, sidelen=img_sidelength,
                                                            max_observations_per_instance=max_observations_per_instance,
                                                            start_idx=start_idx)
        else:
            self.all_instances = get_instance_datasets(data_root, max_num_instances=max_num_instances,
                                                       specific_observation_idcs=specific_observation_idcs,
                                                       cache=cache, sidelen=img_sidelength,
                                                       max_observations_per_instance=max_observations_per_instance,
                                                       start_idx=start_idx)

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def sparsify(self, dict, sparsity):
        if sparsity is None:
            return dict
        else:
            new_dict = {}
            rand_idcs = np.random.choice(self.img_sidelength ** 2, size=sparsity, replace=False)
            for key in ['rgb', 'uv']:
                new_dict[key] = dict[key][rand_idcs]

            for key, v in dict.items():
                if key not in ['rgb', 'uv']:
                    new_dict[key] = dict[key]

            return new_dict

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
        result = defaultdict(list)
        if not batch_list:
            return result

        keys = batch_list[0].keys()

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

        if self.vary_context_number and self.num_context > 0:
            num_context = np.random.randint(1, self.num_context + 1)

        if not self.test:
            try:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=False,
                                               size=self.num_context + self.num_trgt)
            except:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=True,
                                               size=self.num_context + self.num_trgt)

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
                sample = self.all_instances[obj_idx][sample_idcs[i + self.num_context]]

            # post_input.append(sample)
            # post_input[-1]['mask'] = torch.Tensor([1.])

            sub_sample = self.sparsify(sample, self.query_sparsity)
            trgt.append(sub_sample)

        # post_input = self.collate_fn(post_input)

        if self.num_context > 0:
            context = self.collate_fn(context)

        trgt = self.collate_fn(trgt)

        return {'context': context, 'query': trgt}, trgt
