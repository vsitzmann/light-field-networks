'''For general notes on Plucker coordinates:
https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf'''
import numpy as np
import torch

from torch.nn import functional as F
import util


def get_ray_origin(cam2world):
    return cam2world[..., :3, 3]


def plucker_embedding(cam2world, uv, intrinsics):
    '''Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    cam2world: (b, 4, 4)
    intrinsics: (b, 4, 4)
    uv: (b, n, 2)'''
    ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)
    cam_pos = get_ray_origin(cam2world)
    cam_pos = cam_pos[..., None, :].expand(list(uv.shape[:-1]) + [3])

    # https://www.euclideanspace.com/maths/geometry/elements/line/plucker/index.htm
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    return plucker


def closest_to_origin(plucker_coord):
    '''Computes the point on a plucker line closest to the origin.'''
    direction = plucker_coord[..., :3]
    moment = plucker_coord[..., 3:]
    return torch.cross(direction, moment, dim=-1)


def plucker_sd(plucker_coord, point_coord):
    '''Computes the signed distance of a point on a line to the point closest to the origin
    (like a local coordinate system on a plucker line)'''
    # Get closest point to origin along plucker line.
    plucker_origin = closest_to_origin(plucker_coord)

    # Compute signed distance: offset times dot product.
    direction = plucker_coord[..., :3]
    diff = point_coord - plucker_origin
    signed_distance = torch.einsum('...j,...j', diff, direction)
    return signed_distance[..., None]


def get_relative_rotation_matrix(vector_1, vector_2):
    "https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d"
    a_plus_b = vector_1 + vector_2
    outer = a_plus_b.unsqueeze(-2) * a_plus_b.unsqueeze(-1)
    dot = torch.einsum('...j,...j', a_plus_b, a_plus_b)[..., None, None]
    R = 2 * outer/dot - torch.eye(3)[None, None, None].cuda()
    return R


def plucker_reciprocal_product(line_1, line_2):
    '''Computes the reciprocal product between plucker coordinates. See:
    https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf'''
    return torch.einsum('...j,...j', line_1[..., :3], line_2[..., 3:]) + \
           torch.einsum('...j,...j', line_2[..., :3], line_1[..., 3:])


def plucker_distance(line_1, line_2):
    '''Computes the distance between the two closest points on lines parameterized as plucker coordinates.'''
    line_1_dir, line_2_dir = torch.broadcast_tensors(line_1[..., :3], line_2[..., :3])
    direction_cross = torch.cross(line_1_dir, line_2_dir, dim=-1)
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    return torch.abs(plucker_reciprocal_product(line_1, line_2))/direction_cross.norm(dim=-1)


def compute_normal_map(x_img, y_img, z, intrinsics):
    cam_coords = lift(x_img, y_img, z, intrinsics)
    cam_coords = util.lin2img(cam_coords)

    shift_left = cam_coords[:, :, 2:, :]
    shift_right = cam_coords[:, :, :-2, :]

    shift_up = cam_coords[:, :, :, 2:]
    shift_down = cam_coords[:, :, :, :-2]

    diff_hor = F.normalize(shift_right - shift_left, dim=1)[:, :, :, 1:-1]
    diff_ver = F.normalize(shift_up - shift_down, dim=1)[:, :, 1:-1, :]

    cross = torch.cross(diff_hor, diff_ver, dim=1)
    return cross


def get_ray_directions_cam(uv, intrinsics):
    '''Translates meshgrid of uv pixel coordinates to normalized directions of rays through these pixels,
    in camera coordinates.
    '''
    x_cam = uv[..., 0]
    y_cam = uv[..., 1]
    z_cam = torch.ones_like(x_cam).cuda()

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=False)  # (batch_size, -1, 4)
    ray_dirs = F.normalize(pixel_points_cam, dim=-1)
    return ray_dirs


def reflect_vector_on_vector(vector_to_reflect, reflection_axis):
    refl = F.normalize(vector_to_reflect.cuda())
    ax = F.normalize(reflection_axis.cuda())

    r = 2 * (ax * refl).sum(dim=1, keepdim=True) * ax - refl
    return r


def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def project(x, y, z, intrinsics):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    return torch.stack((x_proj, y_proj, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)
    world_coords = torch.einsum('b...ij,b...kj->b...ki', cam2world, pixel_points_cam)[..., :3]

    return world_coords


def project_point_on_line(projection_point, line_direction, point_on_line):
    dot = torch.einsum('...j,...j', projection_point-point_on_line, line_direction)
    return point_on_line + dot[..., None] * line_direction


def get_ray_directions(xy, cam2world, intrinsics):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs


def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).cuda()),
                           dim=2)  # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(cam2world).bmm(points_hom)  # (batch, 4, num_samples)
    depth = points_cam[:, 2, :][:, :, None]  # (batch, num_samples, 1)
    return depth


def ray_sphere_intersect(ray_origin, ray_dir, sphere_center=None, radius=1):
    if sphere_center is None:
        sphere_center = torch.zeros_like(ray_origin)

    ray_dir_dot_origin = torch.einsum('b...jd,b...id->b...ji', ray_dir, ray_origin - sphere_center)
    discrim = torch.sqrt( ray_dir_dot_origin**2 - (torch.einsum('b...id,b...id->b...i', ray_origin-sphere_center, ray_origin - sphere_center)[..., None] - radius**2) )

    t0 = - ray_dir_dot_origin + discrim
    t1 = - ray_dir_dot_origin - discrim
    return ray_origin + t0*ray_dir, ray_origin + t1*ray_dir