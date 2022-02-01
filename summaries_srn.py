import matplotlib
matplotlib.use('Agg')

import torch
import util
import torchvision
import math


def img_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    predictions = model_output['rgb']
    trgt_imgs = ground_truth['rgb']
    indices = model_input['query']['instance_idx']
    depth = model_output['depth']
    normal = model_output['normals']

    predictions = util.flatten_first_two(predictions)
    trgt_imgs = util.flatten_first_two(trgt_imgs)
    normal = util.flatten_first_two(normal)
    depth = util.flatten_first_two(depth)


    with torch.no_grad():
        if 'context' in model_input and model_input['context']:
            context_images = model_input['context']['rgb'] * model_input['context']['mask'][..., None]
            context_images = util.lin2img(util.flatten_first_two(context_images), image_resolution=img_shape)
            writer.add_image(prefix + "context_images",
                             torchvision.utils.make_grid(context_images, scale_each=False, normalize=True).cpu().numpy(),
                             iter)

        output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
        output_vs_gt = util.lin2img(output_vs_gt, image_resolution=img_shape)
        print(output_vs_gt.size(),flush=True)
        writer.add_image(prefix + "output_vs_gt",
                         torchvision.utils.make_grid(output_vs_gt, scale_each=False,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)

        writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        writer.add_scalar(prefix + "idx_min", indices.min(), iter)
        writer.add_scalar(prefix + "idx_max", indices.max(), iter)
        
        depth_maps_plot = util.lin2img(depth)
        writer.add_image(prefix + "pred_depth",
                         torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                                     scale_each=True,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)
        
        normal_plot = util.lin2img(normal.permute(0, -1, 1))
        ns = normal.size()
        normal.resize_(ns[0], ns[1], int(math.sqrt(ns[2])), int(math.sqrt(ns[2])))
        print('SUMMARY', flush=True)
        print(normal.size(), flush=True)
        writer.add_image(prefix + "pred_normals",
                         torchvision.utils.make_grid(normal,
                                                     scale_each=True,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)
        
        
