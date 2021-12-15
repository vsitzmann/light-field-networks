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
####### update imports last


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


class Raymarcher(nn.Module):
    def __init__(self,
                 num_feature_channels,
                 raymarch_steps):
        super().__init__()

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps

        hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)

        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer = nn.Linear(hidden_size, 1)
        self.counter = 0

    def forward(self,
                cam2world,
                phi,
                uv,
                intrinsics):
        batch_size, num_samples, _ = uv.shape

        ray_dirs = geometry.get_ray_directions(uv,
                                               cam2world=cam2world,
                                               intrinsics=intrinsics)

        initial_depth = torch.zeros((batch_size, num_samples)).normal_(mean=0.05, std=5e-4).cuda()
        init_world_coords = geometry.world_from_xy_depth(uv,
                                                         initial_depth,
                                                         intrinsics=intrinsics,
                                                         cam2world=cam2world)
        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]

        for step in range(self.steps):
            v = phi(world_coords[-1])

            state = self.lstm(v.view(-1, self.n_feature_channels), states[-1])

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(batch_size, num_samples, 1)
            new_world_coords = world_coords[-1] + ray_dirs * signed_distance

            states.append(state)
            world_coords.append(new_world_coords)

            depth = geometry.depth_from_world(world_coords[-1], cam2world)

            if self.training:
                print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
                      (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)
        self.counter += 1

        return world_coords[-1], depths[-1]


class SRNsModel(nn.Module):
    def __init__(self, latent_dim, tracing_steps=10, network='relu',
                 fit_single=False, conditioning='hyper'):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 256
        self.fit_single = fit_single
        self.conditioning = conditioning
        self.sphere_trace_steps = tracing_steps
        
        # from SRN repo
#         self.phi_layers = 4  # includes the in and out layers
#         self.rendering_layers = 5  # includes the in and out layers
#         self.sphere_trace_steps = tracing_steps
#         self.freeze_networks = freeze_networks
#         self.fit_single_srn = fit_single_srn
                
        if self.fit_single or conditioning in ['hyper', 'low_rank']:
            if network == 'relu':
                self.phi = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi, 
                                                 num_hidden_layers=2,
                                                 in_features=3, 
                                                 out_features=self.num_hidden_units_phi, 
                                                 outermost_linear=True, 
                                                 norm='layernorm_na')
            elif network == 'siren':
                omega_0 = 30.
                self.phi = custom_layers.Siren(in_features=3, 
                                               hidden_features=256, 
                                               hidden_layers=2,
                                               out_features=out_channels, 
                                               outermost_linear=True, 
                                               hidden_omega_0=omega_0,
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
                nn.Linear(self.num_hidden_units_phi, 3))
            
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

        self.ray_marcher = Raymarcher(num_feature_channels=self.num_hidden_units_phi,
                                                    raymarch_steps=self.sphere_trace_steps)


        self.pixel_generator = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                     num_hidden_layers=1,
                                                     in_features=self.num_hidden_units_phi,
                                                     out_features=3,
                                                     outermost_linear=True)

        print(self)
        print("Number of parameters:")
        util.print_network(self)
        
    def get_srn_function(self, z=None): # can copy
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

    def get_query_cam(self, input): # can copy
        query_dict = input['query']
        pose = util.flatten_first_two(query_dict["cam2world"])
        intrinsics = util.flatten_first_two(query_dict["intrinsics"])
        uv = util.flatten_first_two(query_dict["uv"].float())
        return pose, intrinsics, uv

    def forward(self, input, z=None):
        out_dict = {}
        query = input['query']
        pose, intrinsics, uv = self.get_query_cam(input)
        
        b, _ = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]
        
        if not self.fit_single:
            if 'z' in input:
                z = input['z']
            else:
                z = self.get_z(input)

            out_dict['z'] = z
            z = z.view(b * n_qry, self.latent_dim)

        phi = self.get_srn_function(z)

        # Raymarch SRN phi along rays defined by camera pose, intrinsics and uv coordinates.
        points_xyz, depth_maps = self.ray_marcher(cam2world=pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       phi=phi)

        # Sapmle phi a last time at the final ray-marched world coordinates.
        v = phi(points_xyz)

        # Translate features at ray-marched world coordinates to RGB colors.
        novel_views = self.pixel_generator(v)

        # Calculate normal map
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, 0].view(batch_size, -1)
            y_cam = uv[:, :, 1].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)

            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            out_dict['normals'] = normals.view(b, n_qry, 3, -1) # TODO does not work with varying numbers of query work
        out_dict['rgb'] = novel_views.view(b, n_qry, -1, 3)
        out_dict['depth'] = depth_maps.view(b, n_qry, -1, 1)
        return out_dict

class SRNAutoDecoder(SRNsModel):
    def __init__(self, latent_dim, num_instances, **kwargs):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.num_instances = num_instances

        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def get_z(self, input, val=False):
        instance_idcs = input['query']["instance_idx"].long()
        z = self.latent_codes(instance_idcs)
        return z


class SRNEncoder(SRNsModel):
    def __init__(self, latent_dim, num_instances, network='relu', conditioning='hyper'):
        super().__init__(latent_dim, conditioning=conditioning, network=network)
        self.encoder = conv_modules.Resnet18(c_dim=latent_dim)

    def get_z(self, input, val=False):
        n_qry = input['query']['uv'].shape[1]
        rgb = util.lin2img(util.flatten_first_two(input['context']['rgb']))
        z = self.encoder(rgb)
        z = z.unsqueeze(1).repeat(1, n_qry, 1)
        z *= 1e-2
        return z

