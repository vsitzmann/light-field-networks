"""
Double-checked that softras split was used via (and by switching "dataset_type" to train).
python test.py --experiment_name=nmr_dummy --checkpoint_path=/om2/user/sitzmann/logs/light_fields/NMR_hyper_1e2_reg_layernorm/64_256_None/checkpoints/model_epoch_0087_iter_250000.pth --data_root=/om2/user/egger/MultiClassSRN/data/NMR_Dataset --max_num_instances=10 --dataset=NMR
Then also checked that the reconstruction checkpoint (nmr_rec) and the final checkpoint from this run had the same parameters (which they did). So, false alarm!
"""
# Enable import from parent package
import torch.nn.functional as F
import sys
import os
import numpy as np
import skimage.measure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import training
import multiclass_dataio
import hdf5_dataio

from collections import defaultdict
from glob import glob
import cv2
import conv_modules
import torch
from torch.utils.data import DataLoader
import models
import configargparse
import config
import util
from pathlib import Path

p = configargparse.ArgumentParser()
p.add_argument('--data_root', type=str, required=True)
p.add_argument('--dataset', type=str, required=True)
p.add_argument('--logging_root', type=str, default=config.results_root)
p.add_argument('--checkpoint_path', required=True)
p.add_argument('--experiment_name', type=str, required=True)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--save_out_first_n', type=int, default=100, help='Only saves images of first n object instances.')
p.add_argument('--img_sidelength', type=int, default=64, required=False)
p.add_argument('--viewlist', type=str, default=None, required=False)

opt = p.parse_args()

state_dict = torch.load(opt.checkpoint_path)
num_instances = state_dict['latent_codes.weight'].shape[0]

if opt.viewlist is not None:
    with open(opt.viewlist, "r") as f:
        tmp = [x.strip().split() for x in f.readlines()]
    viewlist = {
        x[0] + "/" + x[1]: list(map(int, x[2:]))
        for x in tmp
    }

model = models.LFAutoDecoder(num_instances=num_instances, latent_dim=256, parameterization='plucker', network=opt.network,
                             conditioning=opt.conditioning).cuda()
model.eval()
print("Loading model")
model.load_state_dict(state_dict)

def convert_image(img, type):
    img = img[0]

    if not 'normal' in type:
        img = util.lin2img(img)[0]
    img = img.cpu().numpy().transpose(1, 2, 0)

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img

def get_psnr(p, trgt):
    p = util.lin2img(p.squeeze(), mode='np')
    trgt = util.lin2img(trgt.squeeze(), mode='np')

    p = util.detach_all(p)
    trgt = util.detach_all(trgt)

    p = (p / 2.) + 0.5
    p = np.clip(p, a_min=0., a_max=1.)
    trgt = (trgt / 2.) + 0.5

    ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
    psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

    return psnr, ssim

print("Loading dataset")
if opt.dataset == 'NMR':
    dataset = multiclass_dataio.get_instance_datasets(opt.data_root, sidelen=opt.img_sidelength, dataset_type='test',
                                                      max_num_instances=opt.max_num_instances)
else:
    dataset = hdf5_dataio.get_instance_datasets_hdf5(opt.data_root, sidelen=opt.img_sidelength,
                                                     max_num_instances=opt.max_num_instances)
log_dir = Path(opt.logging_root) / opt.experiment_name

if opt.dataset == 'NMR':
    class_psnrs = defaultdict(list)
    class_counter = defaultdict(int)
else:
    psnrs = []

with torch.no_grad():
    for i in range(len(dataset)):
        print(f"Object {i:04d}")

        dummy_query = dataset[i][0]
        instance_name = dummy_query['instance_name']

        if opt.dataset == 'NMR':
            obj_class = int(dummy_query['class'].cpu().numpy())
            obj_class = multiclass_dataio.class2string_dict[obj_class]

            if class_counter[obj_class] < opt.save_out_first_n:
                instance_dir = log_dir / f'{obj_class}' / f'{instance_name}'
                instance_dir.mkdir(exist_ok=True, parents=True)
        elif i < opt.save_out_first_n:
            instance_dir = log_dir / f'{instance_name}'
            instance_dir.mkdir(exist_ok=True, parents=True)

        for j, query in enumerate(dataset[i]):
            model_input = util.assemble_model_input(query, query)
            model_output = model(model_input)

            out_dict = {}
            out_dict['rgb'] = model_output['rgb']
            out_dict['gt_rgb'] = model_input['query']['rgb']

            is_context = False
            if opt.viewlist is not None:
                key = '/'.join((obj_class, instance_name))
                if key in viewlist:
                    if j in viewlist[key]:
                        is_context = True
                else:
                    print(f'{key} not in viewlist')
                    continue

            # if opt.dataset != 'NMR' or not is_context:
            psnr, ssim = get_psnr(out_dict['rgb'], out_dict['gt_rgb'])
            if opt.dataset == 'NMR':
                if not is_context:
                    class_psnrs[obj_class].append((psnr, ssim))
            else:
                psnrs.append((psnr, ssim))

            if opt.dataset=='NMR' and class_counter[obj_class] < opt.save_out_first_n:
                for k, v in out_dict.items():
                    img = convert_image(v, k)
                    if k == 'gt_rgb':
                        cv2.imwrite(str(instance_dir / f"{j:06d}_{k}.png"), img)
                    elif k == 'rgb':
                        cv2.imwrite(str(instance_dir / f"{j:06d}.png"), img)
            elif i < opt.save_out_first_n:
                img = convert_image(out_dict['gt_rgb'], 'rgb')
                cv2.imwrite(str(instance_dir / f"{j:06d}_gt.png"), img)
                img = convert_image(out_dict['rgb'], 'rgb')
                cv2.imwrite(str(instance_dir / f"{j:06d}.png"), img)

        if opt.dataset == 'NMR':
            mean_dict = {}
            for k, v in class_psnrs.items():
                mean = np.mean(np.array(v), axis=0)
                mean_dict[k] = f"{mean[0]:.3f} {mean[1]:.3f}"
            print(mean_dict)

            class_counter[obj_class] += 1
        else:
            print(np.mean(np.array(psnrs), axis=0))

with open(os.path.join(log_dir, "results.txt"), "w") as out_file:
    if opt.dataset == 'NMR':
        out_file.write(' & '.join(class_psnrs.keys()) + '\n')

        psnrs, ssims = [], []
        for value in class_psnrs.values():
            mean = np.mean(np.array(value), axis=0)
            psnrs.append(mean[0])
            ssims.append(mean[1])

        out_file.write(' & '.join(map(lambda x: f"{x:.3f}", psnrs)) + '\n')
        out_file.write(' & '.join(map(lambda x: f"{x:.3f}", ssims)) + '\n')
    else:
        mean = np.mean(psnrs, axis=0)
        out_file.write(f"{mean[0]} PSRN {mean[1]} SSIM")
