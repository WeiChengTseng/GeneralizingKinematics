import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import magic.mixture.mdn as mdn
from magic.mixture.utils import *
from magic.reconstruction import calibrations as calibrations
from magic.reconstruction.reconstructor import Reconstructor


def pointcloud_error_metric(depth,
                            trainloader,
                            labels,
                            pi,
                            sigma,
                            mu,
                            obj,
                            ndof,
                            real=False):

    n_gaussians = 20

    # set up reconstructor, transforms
    # proj_matrix = calibrations.sim_proj_matrix
    proj_matrix = calibrations.color_cam_matrix
    recon = Reconstructor(proj_matrix, invalid_val=1000)
    T_magic_to_cam = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                               [1., 0., 0., 0.], [0., 0., 0., 1.0]])

    if real:
        # only labeled refrigerator door, so swap in and out of 2 dof mode
        labels = labels.view(-1, ndof, 8)
        axes = labels[:, :, :7].view(-1, ndof, 7)
        configs = labels[:, :, 7].view(-1, ndof, 1)
        ground_truth_params = {
            'axis': axes,
            'config': configs,
            'radius': torch.zeros(len(axes), ndof, 3)
        }

    else:
        labels = expand_labels(trainloader.dataset.full_labels[:len(labels)],
                               labels, trainloader.dataset.keep_columns,
                               trainloader.dataset.one_columns)
        ground_truth_params = convert_dict_to_real(
            interpret_labels(labels, ndof), trainloader.dataset.bounds, ndof)

    # get most likely shapes, orientations
    gauss_idx = pi.argmax(dim=1)

    # take mean of corresponding gaussian
    output = torch.zeros(mu.shape[0], mu.shape[-1])
    for k in range(len(mu)):
        output[k] = mu[k, gauss_idx[k]]

    # EXPAND
    output = expand_labels(torch.zeros(len(output), 22 if ndof == 1 else 33),
                           output, trainloader.dataset.keep_columns,
                           trainloader.dataset.one_columns)

    # interpret and convert to real.
    param_dict = convert_dict_to_real(interpret_labels(output, ndof),
                                      trainloader.dataset.bounds, ndof)
    real_axis = ground_truth_params['axis']
    real_net_axis = param_dict['axis']

    # get outputs for each of the points using pointcloud candidates
    for i in range(len(depth)):
        x = depth[i]

        ## create pointcloud (in camera frame)
        pcd = recon.point_cloud(x)
        points = pcd.reshape(-1, 3)
        # filtered_points = np.array([p for p in points if not np.isnan(p[2])])
        filtered_points = points
        n_pointcloud_pts = len(filtered_points)

        ## transform pointcloud into magic frame
        transformed_pcd = recon.transform(filtered_points,
                                          np.linalg.inv(T_magic_to_cam))
        pcd_tensor = torch.tensor(transformed_pcd).float()

        ## normalize the columns of the pcd tensor
        for l in range(3):
            bdim = l + 5
            dmax, dmin = trainloader.dataset.bounds[
                bdim, 1], trainloader.dataset.bounds[bdim, 0]
            pcd_tensor[:, l] = (pcd_tensor[:, l] - dmin) / (dmax - dmin)

        for dof in range(ndof):

            # create inputs to gaussian probability
            start_idx = 5 * dof
            end_idx = start_idx + 3

            sigmas = sigma[i, :, start_idx:end_idx].unsqueeze(0).expand(
                len(pcd_tensor), -1, -1)
            mus = mu[i, :, start_idx:end_idx].unsqueeze(0).expand(
                len(pcd_tensor), -1, -1)

            # compute the gaussian probability for each point
            # (using only first 3 columns of mu and sigma)
            likelihoods = mdn.gaussian_probability(sigmas, mus, pcd_tensor)

            probs = torch.sum(likelihoods * pi[i], dim=1)
            prob_img = probs.numpy().reshape(x.shape[0], x.shape[1])

            # take maximum likelihood point from the non-normalized pointcloud.
            best_index = probs.argmax()
            best_point = transformed_pcd[best_index]
            real_net_axis[i, dof, :3] = torch.tensor(best_point)

            # depth[0,0]=12
            # plt.figure()
            # plt.imshow(x, cmap='gray')
            # plt.imshow(prob_img, cmap='hot', alpha=0.3); plt.colorbar()
            # plt.show()

    # position errors
    euclids = euclidean_distance(real_axis[:, :, :3],
                                 real_net_axis[:, :, :3]) * 100.0

    # rotation errors
    axis_quaternions = real_axis[:, :, 3:]
    net_axis_quaternions = real_net_axis.view(-1, ndof, 7)[:, :, 3:]
    axis_angles = quats_to_angles(axis_quaternions)
    net_axis_angles = quats_to_angles(net_axis_quaternions)
    axis_rot_errors = (axis_angles - net_axis_angles).abs() * 180.0 / 3.14159

    # radius errors
    rad_errors = euclidean_distance(ground_truth_params['radius'],
                                    param_dict['radius']) * 100

    # configuration errors
    true_configs = ground_truth_params['config']
    net_configs = param_dict['config']

    config_diff = (true_configs - net_configs).abs()
    config_errors = config_diff.view(-1, ndof)
    if obj == 'drawer':
        config_errors = config_errors * 100  # convert to cm for drawers
    else:
        config_errors = config_errors * 180.0 / 3.14159  # convert to degrees for revolutes

    # refrigerator, take only the first label.
    if real and obj == 'refrigerator':
        euclids = euclids[:, 0].view(-1, 1)
        axis_rot_errors = axis_rot_errors[:, 0].view(-1, 1)
        rad_errors = rad_errors[:, 0].view(-1, 1)
        config_errors = config_errors[:, 0].view(-1, 1)

    return euclids.mean(dim=1), axis_rot_errors.mean(dim=1), rad_errors.mean(
        dim=1), config_errors.mean(dim=1)


def max_likelihood_error(trainloader,
                         labels,
                         pi,
                         sigma,
                         mu,
                         obj,
                         ndof,
                         real=False):
    if real:

        # only labeled refrigerator door, so swap in and out of 2 dof mode
        labels = labels.view(-1, ndof, 8)
        axes = labels[:, :, :7].view(-1, ndof, 7)
        configs = labels[:, :, 7].view(-1, ndof, 1)
        ground_truth_params = {
            'axis': axes,
            'config': configs,
            'radius': torch.zeros(len(axes), ndof, 3)
        }

    else:
        labels = expand_labels(trainloader.dataset.full_labels[:len(labels)],
                               labels, trainloader.dataset.keep_columns,
                               trainloader.dataset.one_columns)
        ground_truth_params = convert_dict_to_real(
            interpret_labels(labels, ndof), trainloader.dataset.bounds, ndof)

    # get most likely gaussian
    gauss_idx = pi.argmax(dim=1)

    # take mean of corresponding gaussian
    output = torch.zeros(mu.shape[0], mu.shape[-1])
    for k in range(len(mu)):
        output[k] = mu[k, gauss_idx[k]]

    output = expand_labels(torch.zeros(len(output), 22 if ndof == 1 else 33),
                           output, trainloader.dataset.keep_columns,
                           trainloader.dataset.one_columns)

    # interpret and convert to real.
    param_dict = convert_dict_to_real(interpret_labels(output, ndof),
                                      trainloader.dataset.bounds, ndof)
    real_axis = ground_truth_params['axis']
    real_net_axis = param_dict['axis']

    # position errors
    euclids = euclidean_distance(real_axis[:, :, :3],
                                 real_net_axis[:, :, :3]) * 100.0

    # rotation errors
    axis_quaternions = real_axis[:, :, 3:]
    net_axis_quaternions = real_net_axis.view(-1, ndof, 7)[:, :, 3:]
    axis_angles = quats_to_angles(axis_quaternions)
    net_axis_angles = quats_to_angles(net_axis_quaternions)
    axis_rot_errors = (axis_angles - net_axis_angles).abs() * 180.0 / 3.14159

    # radius errors
    rad_errors = euclidean_distance(ground_truth_params['radius'],
                                    param_dict['radius']) * 100

    # configuration errors
    true_configs = ground_truth_params['config']
    net_configs = param_dict['config']

    config_diff = (true_configs - net_configs).abs()
    config_errors = config_diff.view(-1, ndof)
    if obj == 'drawer':
        config_errors = config_errors * 100  # convert to cm for drawers
    else:
        config_errors = config_errors * 180.0 / 3.14159  # convert to degrees for revolutes

    # refrigerator, take only the first label.
    if real and obj == 'refrigerator':
        euclids = euclids[:, 0].view(-1, 1)
        axis_rot_errors = axis_rot_errors[:, 0].view(-1, 1)
        rad_errors = rad_errors[:, 0].view(-1, 1)
        config_errors = config_errors[:, 0].view(-1, 1)

    return euclids.mean(dim=1), axis_rot_errors.mean(dim=1), rad_errors.mean(
        dim=1), config_errors.mean(dim=1)


def mean_sample_error(trainloader,
                      labels,
                      pi,
                      sigma,
                      mu,
                      obj,
                      ndof,
                      real=False):
    '''
	Make error table for the labels and MDN params given.
	In practice, used for just one batch element.
	'''
    # draw samples from MoG defined by parameters
    output = mdn.sample(pi, sigma, mu)

    # expand labels and output back to full size
    output = expand_labels(torch.zeros(len(output), 22 if ndof == 1 else 33),
                           output, trainloader.dataset.keep_columns,
                           trainloader.dataset.one_columns)

    if real:
        labels = labels.view(-1, 1, 8)
        axes = labels[:, :, :7]
        configs = labels[:, :, 7]
        ground_truth_params = {
            'axis': axes,
            'config': configs,
            'radius': torch.zeros(len(axes), ndof, 3)
        }

    else:
        labels = expand_labels(trainloader.dataset.full_labels[:len(labels)],
                               labels, trainloader.dataset.keep_columns,
                               trainloader.dataset.one_columns)
        ground_truth_params = convert_dict_to_real(
            interpret_labels(labels, ndof), trainloader.dataset.bounds, ndof)

    # interpret and convert to real.
    param_dict = convert_dict_to_real(interpret_labels(output, ndof),
                                      trainloader.dataset.bounds, ndof)
    real_axis = ground_truth_params['axis']
    real_net_axis = param_dict['axis']

    # position errors
    euclids = euclidean_distance(real_axis[:, :, :3],
                                 real_net_axis[:, :, :3]) * 100.0

    # rotation errors
    axis_quaternions = real_axis[:, :, 3:]
    net_axis_quaternions = real_net_axis.view(-1, ndof, 7)[:, :, 3:]
    axis_angles = quats_to_angles(axis_quaternions)
    net_axis_angles = quats_to_angles(net_axis_quaternions)
    axis_rot_errors = (axis_angles - net_axis_angles).abs() * 180.0 / 3.14159

    # radius errors
    rad_errors = euclidean_distance(ground_truth_params['radius'],
                                    param_dict['radius']) * 100

    # configuration errors
    true_configs = ground_truth_params['config']
    net_configs = param_dict['config']
    config_errors = (true_configs - net_configs).abs()
    if obj == 'drawer':
        config_errors = config_errors * 100  # convert to cm for drawers
    else:
        config_errors = config_errors * 180.0 / 3.14159  # convert to degrees for revolutes

    return euclids.mean(), axis_rot_errors.mean(), rad_errors.mean(
    ), config_errors.mean()


def mixture_error_table(name,
                        model,
                        dataloader,
                        device,
                        n_gaussians,
                        obj,
                        ndof,
                        normalize=True,
                        real=False,
                        sample_error=False,
                        pointcloud_error=False):
    ''' Make the error table for the model. '''

    print('computing %s /euclid/ histograms.' % name)

    # make plot directory
    if not os.path.exists("plots/" + name + "/euclid_hist"):
        os.makedirs("plots/" + name + "/euclid_hist")

    angle_len = 1
    label_dim = 7 + 1 + 3  # axis(x,y,z,qw,qx,qy,qz), door angle(rad), radius (m)
    euclids = torch.zeros(len(dataloader.dataset))
    axis_rot_errors = torch.zeros_like(euclids)
    radius_errors = torch.zeros_like(euclids)
    config_errors = torch.zeros_like(euclids)

    N_samples = 100
    with torch.no_grad():
        k = 0
        for i, X in enumerate(tqdm(dataloader)):

            # pass batch through network
            depth, labels = X['depth'].to(device), X['label'].to(device)

            pi, sigma, mu = model(depth)
            #
            # import matplotlib.pyplot as plt
            # plt.imshow(depth[0,0])
            # plt.show()

            if sample_error:

                # for each batch element, compute mean sample error
                for j in range(len(pi)):

                    euc_i, rot_i, rad_i, conf_i = mean_sample_error(
                        dataloader,
                        labels[j].view(1, -1).expand(N_samples, -1),
                        pi[j].view(1, -1).expand(N_samples, -1),
                        sigma[j].view(1, n_gaussians,
                                      -1).expand(N_samples, -1, -1),
                        mu[j].view(1, n_gaussians,
                                   -1).expand(N_samples, -1, -1),
                        obj,
                        ndof,
                        real=real)
                    # store the mean sample error here
                    euclids[k] = euc_i
                    axis_rot_errors[k] = rot_i
                    radius_errors[k] = rad_i
                    config_errors[k] = conf_i

                    # increment the element index
                    k += 1

            elif pointcloud_error:
                og_depth = X['og_depth']
                euc_i, rot_i, rad_i, conf_i = pointcloud_error_metric(
                    og_depth,
                    dataloader,
                    labels,
                    pi,
                    sigma,
                    mu,
                    obj,
                    ndof,
                    real=real)
                # assign each of these to their respective arrays
                euclids[i * dataloader.batch_size:i * dataloader.batch_size +
                        len(euc_i)] = euc_i
                axis_rot_errors[i * dataloader.batch_size:i *
                                dataloader.batch_size + len(euc_i)] = rot_i
                radius_errors[i *
                              dataloader.batch_size:i * dataloader.batch_size +
                              len(euc_i)] = rad_i
                config_errors[i *
                              dataloader.batch_size:i * dataloader.batch_size +
                              len(euc_i)] = conf_i

            else:

                euc_i, rot_i, rad_i, conf_i = max_likelihood_error(dataloader,
                                                                   labels,
                                                                   pi,
                                                                   sigma,
                                                                   mu,
                                                                   obj,
                                                                   ndof,
                                                                   real=real)

                # assign each of these to their respective arrays
                euclids[i * dataloader.batch_size:i * dataloader.batch_size +
                        len(euc_i)] = euc_i
                axis_rot_errors[i * dataloader.batch_size:i *
                                dataloader.batch_size + len(euc_i)] = rot_i
                radius_errors[i *
                              dataloader.batch_size:i * dataloader.batch_size +
                              len(euc_i)] = rad_i
                config_errors[i *
                              dataloader.batch_size:i * dataloader.batch_size +
                              len(euc_i)] = conf_i

    # plt.hist(euclids, bins=32); plt.show()

    # make std deviation table
    table = torch.tensor([
        euclids.mean(),
        euclids.std(),
        axis_rot_errors.mean(),
        axis_rot_errors.std(),
        radius_errors.mean(),
        radius_errors.std(),
        config_errors.mean(),
        config_errors.std()
    ]).view(1, 8).numpy()

    df = pd.DataFrame(table)
    fpath = 'plots/%s/table.xlsx' % name
    df.to_excel(fpath, index=False)
    return table
