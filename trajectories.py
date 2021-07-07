import torch
import torch.nn as nn
import numpy as np
import unittest

def look_at_deepvoxels(translation, point):
    '''Look at function that conforms with DeepVoxels, DeepSpace and DeepClouds coordinate system: Camera
    looks in negative z direction.

    :param translation:
    :param point:
    :return:
    '''
    target_pose = np.zeros((4,4))
    target_pose[:3,3] = translation

    direction = point - translation

    dir_norm = direction / np.linalg.norm(direction)

    tmp = np.array([0.,1.,0.])
    right_vector = np.cross(tmp,-1. * dir_norm) # Camera points in negative z-direction
    right_vector /= np.linalg.norm(right_vector)

    up_vector = np.cross(dir_norm, right_vector)

    target_pose[:3,2] = dir_norm
    target_pose[:3,1] = up_vector
    target_pose[:3,0] = right_vector

    target_pose[3,3] = 1.

    return target_pose


def look_at_cars(translation, point):
    '''Look at function that conforms with DeepVoxels, DeepSpace and DeepClouds coordinate system: Camera
    looks in negative z direction.

    :param translation:
    :param point:
    :return:
    '''
    target_pose = np.zeros((4,4))
    target_pose[:3,3] = translation

    direction = point - translation

    dir_norm = direction / np.linalg.norm(direction)

    tmp = np.array([0.,0.,1.])
    right_vector = np.cross(tmp,-1. * dir_norm) # Camera points in negative z-direction
    right_vector /= np.linalg.norm(right_vector)

    up_vector = np.cross(dir_norm, right_vector)

    target_pose[:3,2] = dir_norm
    target_pose[:3,1] = up_vector
    target_pose[:3,0] = right_vector

    target_pose[3,3] = 1.

    return target_pose


def look_at_rooms(translation, point):
    '''Look at function that conforms with DeepVoxels, DeepSpace and DeepClouds coordinate system: Camera
    looks in negative z direction.

    :param translation:
    :param point:
    :return:
    '''
    target_pose = np.zeros((4,4))
    target_pose[:3,3] = translation

    direction = point - translation

    dir_norm = direction / np.linalg.norm(direction)

    tmp = np.array([0., 1., 0.])
    right_vector = np.cross(tmp, dir_norm) # Camera points in negative z-direction
    right_vector /= np.linalg.norm(right_vector)

    up_vector = np.cross(dir_norm, right_vector)

    target_pose[:3,2] = dir_norm
    target_pose[:3,1] = up_vector
    target_pose[:3,0] = right_vector

    target_pose[3,3] = 1.

    return target_pose


def rooms_360(look_at_fn, radius=1, num_samples=200, altitude=45):
    trajectory = []
    virtual_radius = np.cos(np.deg2rad(altitude)) * radius

    for angle in np.linspace(0, 2*np.pi, num_samples):
        translation = np.array([virtual_radius*np.sin(angle),
                                0.75,
                                virtual_radius*np.cos(angle)])
        look_at = 2*radius*translation
        look_at[1] = 0.75
        cam2world = look_at_fn(translation, look_at)
        cam2world = torch.from_numpy(cam2world).float()
        cam2world[1, 3] = 0.75
        cam2world[:3, 1] = torch.Tensor([0., -1., 0.])
        trajectory.append(cam2world)

    return trajectory


def around(look_at_fn, radius=1, num_samples=200, altitude=45):
    '''

    :param radius:
    :param num_samples:
    :param altitude: Altitude in degree.
    :return:
    '''
    trajectory = []

    z_coord = np.sin(np.deg2rad(altitude)) * radius
    virtual_radius = np.cos(np.deg2rad(altitude)) * radius

    for angle in np.linspace(0, 2*np.pi, num_samples):
        translation = np.array([virtual_radius*np.sin(angle),
                                virtual_radius*np.cos(angle),
                                z_coord])
        cam2world = look_at_fn(translation, np.array([0.,0.,0.]))
        cam2world = torch.from_numpy(cam2world).float()
        trajectory.append(cam2world)

    return trajectory


def back_and_forth(look_at_fn, radius=1, num_samples=200, altitude=0):
    '''

    :param radius:
    :param num_samples:
    :param altitude: Altitude in degree.
    :return:
    '''
    trajectory = []

    z_coord = np.sin(np.deg2rad(altitude)) * radius
    virtual_radius = np.cos(np.deg2rad(altitude)) * radius

    distances = np.linspace(1, 5, num_samples)
    distances = np.concatenate((distances, distances[::-1]), axis=-1)

    for distance in distances:
        translation = np.array([virtual_radius*np.sin(0),
                                virtual_radius*np.cos(0),
                                z_coord]) * distance
        cam2world = look_at_fn(translation, np.array([0.,0.,0.]))
        cam2world = torch.from_numpy(cam2world).float()
        trajectory.append(cam2world)

    return trajectory
