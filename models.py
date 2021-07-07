import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np

import util

import conv_modules
import custom_layers
import geometry
import hyperlayers


class LightFieldModel(nn.Module):
    def __init__(self, latent_dim, parameterization='plucker', network='relu',
                 fit_single=False, conditioning='hyper', depth=False, alpha=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 256
        self.fit_single = fit_single
        self.parameterization = parameterization
        self.conditioning = conditioning
        self.depth = depth
        self.alpha = alpha

        out_channels = 3

        if self.depth:
            out_channels += 1
        if self.alpha:
            out_channels += 1
            self.background = torch.ones((1, 1, 1, 3)).cuda()

        if self.fit_single or conditioning in ['hyper', 'low_rank']:
            if network == 'relu':
                self.phi = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=6,
                                                 in_features=6, out_features=out_channels, outermost_linear=True, norm='layernorm_na')
            elif network == 'siren':
                # if self.fit_single:
                omega_0 = 30.
                # else:
                #     omega_0 = 1.
                self.phi = custom_layers.Siren(in_features=6, hidden_features=256, hidden_layers=8,
                                               out_features=out_channels, outermost_linear=True, hidden_omega_0=omega_0,
                                               first_omega_0=omega_0)
        elif conditioning == 'concat':
            self.phi = nn.Sequential(
                nn.Linear(6+self.latent_dim, self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                nn.Linear(self.num_hidden_units_phi, 3)
            )

        if not self.fit_single:
            if conditioning=='hyper':
                self.hyper_phi = hyperlayers.HyperNetwork(hyper_in_features=self.latent_dim,
                                                          hyper_hidden_layers=1,
                                                          hyper_hidden_features=self.latent_dim,
                                                          hypo_module=self.phi)
            elif conditioning=='low_rank':
                self.hyper_phi = hyperlayers.LowRankHyperNetwork(hyper_in_features=self.latent_dim,
                                                                 hyper_hidden_layers=1,
                                                                 hyper_hidden_features=512,
                                                                 hypo_module=self.phi,
                                                                 nonlinearity='leaky_relu')

        print(self.phi)
        print(np.sum(np.prod(param.shape) for param in self.phi.parameters()))

    def get_light_field_function(self, z=None):
        if self.fit_single:
            phi = self.phi
        elif self.conditioning in ['hyper', 'low_rank']:
            phi_weights = self.hyper_phi(z)
            phi = lambda x: self.phi(x, params=phi_weights)
        elif self.conditioning == 'concat':
            def phi(x):
                b, n_pix = x.shape[:2]
                z_rep = z.view(b, 1, self.latent_dim).repeat(1, n_pix, 1)
                return self.phi(torch.cat((z_rep, x), dim=-1))
        return phi

    def get_query_cam(self, input):
        query_dict = input['query']
        pose = util.flatten_first_two(query_dict["cam2world"])
        intrinsics = util.flatten_first_two(query_dict["intrinsics"])
        uv = util.flatten_first_two(query_dict["uv"].float())
        return pose, intrinsics, uv

    def forward(self, input, val=False, compute_depth=False, timing=False):
        out_dict = {}
        query = input['query']
        b, n_ctxt = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]

        if not self.fit_single:
            if 'z' in input:
                z = input['z']
            else:
                z = self.get_z(input)

            out_dict['z'] = z
            z = z.view(b * n_qry, self.latent_dim)

        query_pose, query_intrinsics, query_uv = self.get_query_cam(input)

        if self.parameterization == 'plucker':
            light_field_coords = geometry.plucker_embedding(query_pose, query_uv, query_intrinsics)
        else:
            ray_origin = query_pose[:, :3, 3][:, None, :]
            ray_dir = geometry.get_ray_directions(query_uv, query_pose, query_intrinsics)
            intsec_1, intsec_2 = geometry.ray_sphere_intersect(ray_origin, ray_dir, radius=100)
            intsec_1 = F.normalize(intsec_1, dim=-1)
            intsec_2 = F.normalize(intsec_2, dim=-1)

            light_field_coords = torch.cat((intsec_1, intsec_2), dim=-1)
            out_dict['intsec_1'] = intsec_1
            out_dict['intsec_2'] = intsec_2
            out_dict['ray_dir'] = ray_dir
            out_dict['ray_origin'] = ray_origin

        light_field_coords.requires_grad_(True)
        out_dict['coords'] = light_field_coords.view(b*n_qry, n_pix, 6)

        lf_function = self.get_light_field_function(None if self.fit_single else z)
        out_dict['lf_function'] = lf_function

        if timing: t0 = time.time()
        lf_out = lf_function(out_dict['coords'])
        if timing: t1 = time.time(); total_n = t1 - t0; print(f'{total_n}')

        rgb = lf_out[..., :3]

        if self.depth:
            depth = lf_out[..., 3:4]
            out_dict['depth'] = depth.view(b, n_qry, n_pix, 1)

        rgb = rgb.view(b, n_qry, n_pix, 3)

        if self.alpha:
            alpha = lf_out[..., -1:].view(b, n_qry, n_pix, 1)
            weight = 1 - torch.exp(-torch.abs(alpha))
            rgb = weight * rgb + (1 - weight) * self.background
            out_dict['alpha'] = weight

        if compute_depth:
            with torch.enable_grad():
                lf_function = self.get_light_field_function(z)
                depth = util.light_field_depth_map(light_field_coords, query_pose, lf_function)['depth']
                depth = depth.view(b, n_qry, n_pix, 1)
                out_dict['depth'] = depth

        out_dict['rgb'] = rgb
        return out_dict


class LFAutoDecoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', **kwargs):
        super().__init__(latent_dim=latent_dim, parameterization=parameterization, **kwargs)
        self.num_instances = num_instances

        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def get_z(self, input, val=False):
        instance_idcs = input['query']["instance_idx"].long()
        z = self.latent_codes(instance_idcs)
        return z


class LFEncoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', conditioning='hyper'):
        super().__init__(latent_dim, parameterization, conditioning='low_rank')
        self.num_instances = num_instances
        self.encoder = conv_modules.Resnet18(c_dim=latent_dim)

    def get_z(self, input, val=False):
        n_qry = input['query']['uv'].shape[1]
        rgb = util.lin2img(util.flatten_first_two(input['context']['rgb']))
        z = self.encoder(rgb)
        z = z.unsqueeze(1).repeat(1, n_qry, 1)
        z *= 1e-2
        return z
