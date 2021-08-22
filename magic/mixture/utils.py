import copy
import torch
import numpy as np


def norm_quat(x):
    # input shape is (-1, ndof, 4)
    # output shape is (-1, ndof, 4), normalized
    norms = torch.norm(x, p=2, dim=2)
    # print(x.shape)
    # print(norms.shape)
    return x / norms.unsqueeze(2).expand_as(x)


def angle_to_quat(angle, axis=[0, 0, 1]):
    qx = axis[0] * np.sin(angle / 2)
    qy = axis[1] * np.sin(angle / 2)
    qz = axis[2] * np.sin(angle / 2)
    qw = np.cos(angle / 2)
    return np.array([qw, qx, qy, qz])


def quats_to_angles(quaternions):
    # new shape is (-1, ndof, 4)
    # new out shape should be (-1, ndof, 1) or (-1, ndof)
    quaternions = norm_quat(quaternions)

    # guard against nan:
    quaternions[:, :, 0] = torch.clamp(quaternions[:, :, 0], 0., 1.)

    out = 2 * torch.acos(quaternions)
    return out[:, :, 0]
    # return torch.tensor([2*torch.acos(q[0]) for q in quaternions])


def euclidean_distance(x, y):
    # note: assuming tensor shapes are (-1, ndof, n)
    # return shape will be (-1,ndof)
    diff_sq = (y - x) * (y - x)
    return torch.sum(diff_sq, 2).sqrt()


def expand_labels(full_labels, filtered_labels, keep_columns, one_columns):
    restored_labels = torch.zeros(
        len(filtered_labels), full_labels.shape[1]).to(filtered_labels.device)
    restored_labels[:, keep_columns] = filtered_labels
    restored_labels[:, one_columns] = torch.ones(len(filtered_labels),
                                                 len(one_columns)).to(
                                                     filtered_labels.device)
    return restored_labels


def interpret_labels(y, n_dof):
    axis_len = 7
    rad_len = 3
    config_len = 1
    shape_len = 4
    pose_len = 7

    axis_start_idx = 0
    config_start_idx = 0 + axis_len * n_dof
    rad_start_idx = config_start_idx + config_len * n_dof
    shape_start_idx = rad_start_idx + rad_len * n_dof
    pose_start_idx = axis_len * n_dof + config_len * n_dof + rad_len * n_dof + shape_len

    # shape and pose are constant, extract now
    shape = y[:, shape_start_idx:shape_start_idx + shape_len]
    pose = y[:, pose_start_idx:pose_start_idx + pose_len]

    # loop over ndof and fill dictionary elements
    axis = torch.zeros(len(y), n_dof, axis_len)
    config = torch.zeros(len(y), n_dof, config_len)
    rad = torch.zeros(len(y), n_dof, rad_len)
    for i in range(n_dof):
        axis[:, i, :] = y[:, i * axis_len:(i + 1) * axis_len]
        config[:,
               i, :] = y[:,
                         config_start_idx + i * config_len:config_start_idx +
                         (i + 1) * config_len]
        rad[:, i, :] = y[:, rad_start_idx + i * rad_len:rad_start_idx +
                         (i + 1) * rad_len]

    return {
        'axis': axis,
        'config': config,
        'radius': rad,
        'geom': shape,
        'pose': pose
    }


def convert_to_real(X, bounds, start_idx):
    y = torch.zeros_like(X, device=X.device)
    for i in range(len(X)):
        x = X[i]
        for j in range(len(x)):
            y[i,
              j] = x[j] * (bounds[start_idx + j, 1] -
                           bounds[start_idx + j, 0]) + bounds[start_idx + j, 0]
    return y


def convert_dict_to_real(label_dict, bounds, n_dof):
    #TODO: 2DoF here, converting appropriately
    # axis = convert_to_real(label_dict['axis'].view(-1,7 * n_dof), bounds, 5)
    # radii = convert_to_real(label_dict['radius'].view(-1,3), bounds,12)
    # angle = convert_to_real(label_dict['config'].view(-1,1),  bounds, 22)
    # shape = convert_to_real(label_dict['geom'].view(-1,4),  bounds, 1)
    # pose = convert_to_real(label_dict['pose'].view(-1,7),  bounds, 15)
    axis_len = 7
    rad_len = 3
    config_len = 1
    shape_len = 4
    pose_len = 7

    axis = convert_to_real(label_dict['axis'].view(-1, 7 * n_dof), bounds,
                           5).view(-1, n_dof, 7)
    radii = convert_to_real(label_dict['radius'].view(-1, 3 * n_dof), bounds,
                            5 + n_dof * axis_len).view(-1, n_dof, 3)
    angle = convert_to_real(label_dict['config'].view(-1, 1 * n_dof), bounds,
                            5 + n_dof * axis_len + n_dof * rad_len +
                            pose_len).view(-1, n_dof, 1)
    shape = convert_to_real(label_dict['geom'].view(-1, 4), bounds, 1)
    pose = convert_to_real(label_dict['pose'].view(-1, 7), bounds,
                           5 + n_dof * axis_len + n_dof * rad_len)

    return {
        'axis': axis,
        'config': angle,
        'radius': radii,
        'geom': shape,
        'pose': pose
    }


def q_to_idx(q):
    differences = torch.abs(
        torch.tensor([
            -3.7259654560926E-06, -0.0619193502438947, -0.158249459384306,
            -0.257895820952111, -0.357861743606078, -0.457858459841061,
            -0.55785814341053, -0.657858112918615, -0.757858109980349,
            -0.857858109697211, -0.957858109669927, -1.0578581096673,
            -1.15785810966704, -1.25785810966702, -1.35785810966701,
            -1.457858109667
        ]) - q)
    return torch.argmin(differences)
