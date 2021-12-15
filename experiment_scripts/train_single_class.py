# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager
import torch
import models
import training
import summaries
import hdf5_dataio
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import config

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default=config.logging_root, required=False, help='root for logging')
p.add_argument('--data_root', type=str, required=True)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--experiment_name', type=str, required=True)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=1000)
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--max_observations_per_instance', type=int, default=None)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)
p.add_argument('--optim_checkpoint_path', default=None)
opt = p.parse_args()

batch_sizes = 256, 50
sidelens = 64, 128

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)


def multigpu_train(gpu, opt, cache):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    def create_dataloader_callback(sidelength, batch_size, query_sparsity):
        train_dataset = hdf5_dataio.SceneClassDataset(num_context=0, num_trgt=opt.num_trgt,
                                                      data_root=opt.data_root, query_sparsity=query_sparsity,
                                                      img_sidelength=sidelength, vary_context_number=True, cache=cache,
                                                      max_num_instances=opt.max_num_instances,
                                                      max_observations_per_instance=opt.max_observations_per_instance)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)
        return train_loader

    num_instances = hdf5_dataio.get_num_instances(opt.data_root)
    model = models.LFAutoDecoder(latent_dim=256, num_instances=num_instances, parameterization='plucker',
                                 network=opt.network, conditioning=opt.conditioning).cuda()

    if opt.checkpoint_path is not None:
        state_dict = torch.load(opt.checkpoint_path)
        model.load_state_dict(state_dict)

    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    summary_fn = summaries.img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = val_loss_fn = loss_functions.LFLoss(reg_weight=1)

    if opt.optim_checkpoint_path is not None:
        optimizers = [torch.optim.Adam(lr=opt.lr, params=model.parameters())]
        state_dict = torch.load(opt.optim_checkpoint_path)
        optimizers[0].load_state_dict(state_dict)
    else:
        optimizers=None

    training.multiscale_training(model=model, dataloader_callback=create_dataloader_callback,
                                 dataloader_iters=(10000, 500000),
                                 dataloader_params=((sidelens[0], batch_sizes[0], None), (sidelens[1], batch_sizes[1], None)),
                                 epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                 epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
                                 iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                                 overwrite=True, optimizers=optimizers,
                                 rank=gpu, train_function=training.train, gpus=opt.gpus)

if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()

    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_dict))
    else:
        multigpu_train(0, opt, shared_dict)
