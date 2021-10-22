'''Implements a generic training loop.
'''

import os
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size



def multiscale_training(train_function, dataloader_callback, dataloader_iters, dataloader_params, **kwargs):
    model = kwargs.pop('model', None)
    optimizers = kwargs.pop('optimizers', None)
    org_model_dir = kwargs.pop('model_dir', None)

    for params, max_steps in zip(dataloader_params, dataloader_iters):
        dataloaders = dataloader_callback(*params)
        model_dir = os.path.join(org_model_dir, '_'.join(map(str, params)))

        model, optimizers = train_function(dataloaders=dataloaders, model_dir=model_dir, model=model,
                                           optimizers=optimizers,
                                           max_steps=max_steps, **kwargs)


def train(model, dataloaders, epochs, lr, epochs_til_checkpoint, model_dir, loss_fn, steps_til_summary=1,
          summary_fn=None, iters_til_checkpoint=None, clip_grad=False, val_loss_fn=None, val_summary_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None,
          loss_schedules=None, device='gpu'):
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if isinstance(dataloaders, tuple):
        train_dataloader, val_dataloader = dataloaders
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"
    else:
        train_dataloader, val_dataloader = dataloaders, None

    if rank==0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir, flush_secs=10)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))

            for step, (model_input, gt) in enumerate(train_dataloader):

                if device == 'gpu':
                    model_input = util.dict_to_gpu(model_input)
                    gt = util.dict_to_gpu(gt)

                model_output = model(model_input)
                losses, loss_summaries = loss_fn(model_output, gt, model=model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if (loss_schedules is not None) and (loss_name in loss_schedules):
                        if rank == 0:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    for i, optim in enumerate(optimizers):
                        torch.save(optim.state_dict(),
                                   os.path.join(checkpoints_dir, f'optim_{i}_current.pth'))
                    if summary_fn is not None:
                        summary_fn(model, model_input, gt, loss_summaries, model_output, writer, total_steps, 'train_')

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()
                del train_loss

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print(", ".join([f"Epoch {epoch}"] + [f"{name} {loss.mean()}" for name, loss in losses.items()]))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                if device == 'gpu':
                                    model_input = util.dict_to_gpu(model_input)
                                    gt = util.dict_to_gpu(gt)

                                model_output = model(model_input, val=True)
                                val_loss, val_loss_smry = val_loss_fn(model_output, gt, val=True, model=model)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value)

                                if val_i == batches_per_validation:
                                    break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(np.concatenate([l.reshape(-1).cpu().numpy() for l in loss], axis=0))

                                if rank == 0:
                                    writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                            if rank == 0:
                                if val_summary_fn is not None:
                                    val_summary_fn(model, model_input, gt, val_loss_smry, model_output, writer, total_steps, 'val_')

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

        if rank == 0:
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))

        return model, optimizers
