import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import magic.mixture.mdn as mdn
from magic.mixture.utils import *


class MDNTrainer(object):
    """ Trains a mixture density network to predict object kinematics  """
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 optimizer,
                 epochs,
                 name,
                 test_freq,
                 device,
                 obj='microwave',
                 ndof=1):
        super(MDNTrainer, self).__init__()
        self.model = model
        self.trainloader = train_loader
        self.testloader = test_loader
        self.optimizer = optimizer
        self.criterion = mdn.mdn_loss
        self.epochs = epochs
        self.name = name
        self.test_freq = test_freq
        self.obj = obj
        self.ndof = ndof

        self.normalize = True
        self.losses = []
        self.tlosses = []

        # float model as push to GPU/CPU
        self.device = device
        self.model.float().to(self.device)

    def train(self):
        best_tloss = 1e8
        for epoch in range(self.epochs):
            sys.stdout.flush()
            loss = self.train_epoch(epoch)
            self.losses.append(loss)
            if epoch % self.test_freq == 0:
                tloss = self.test_epoch(epoch)
                self.tlosses.append(tloss)
                self.plot_losses()

                if tloss < best_tloss:
                    print('saving model.')
                    net_fname = 'models/' + str(self.name) + '.net'
                    torch.save(self.model.state_dict(), net_fname)
                    best_tloss = tloss

        # plot losses one more time
        self.plot_losses()

        # re-load the best state dictionary that was saved earlier.
        self.model.load_state_dict(torch.load(net_fname, map_location='cpu'))

        return self.model

    def train_epoch(self, epoch):
        # print('train epoch')
        start = time.time()
        running_loss = 0
        batches_per_dataset = len(
            self.trainloader.dataset) / self.trainloader.batch_size
        for i, X in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            depth, labels = X['depth'].to(self.device), X['label'].to(
                self.device)

            #
            # depth[0,0,0,0]=1.0
            # plt.imshow(depth[0,0].numpy())
            # plt.show()

            pi, sigma, mu = self.model(depth)
            loss = self.criterion(pi, sigma, mu, labels)
            if loss.data == -float('inf'):
                print('inf loss caught, not backpropping')
                running_loss += -1000
            else:
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        euc = self.analyze_shape_loss(labels, pi, sigma, mu)
        # euc = torch.tensor([1,2,3]).float()
        stop = time.time()
        print(
            'Epoch %s -  Train  Loss: %.5f Euc. Axis Error: %.3f Time: %.5f' %
            (str(epoch).zfill(3), running_loss / batches_per_dataset,
             euc.mean(), stop - start))
        return running_loss / batches_per_dataset

    def test_epoch(self, epoch):
        # print('test epoch')
        start = time.time()
        running_loss = 0
        batches_per_dataset = len(
            self.testloader.dataset) / self.testloader.batch_size
        with torch.no_grad():
            for i, X in enumerate(self.testloader):
                depth, labels = X['depth'].to(self.device), X['label'].to(
                    self.device)
                pi, sigma, mu = self.model(depth)
                loss = self.criterion(pi, sigma, mu, labels)
                running_loss += loss.item()

        # print('pi', pi.shape)
        # print('sigma', sigma.shape)
        # print('mu', mu.shape)
        euc = self.analyze_shape_loss(labels, pi, sigma, mu)
        # try:
        # 	euc = self.analyze_shape_loss(labels, pi, sigma, mu)
        # except Exception as e:
        # 	print('cuda sampling error')
        # 	euc=torch.tensor([-1]).float()

        stop = time.time()
        print('Epoch %s -  Test  Loss: %.5f Euc. Axis Error: %.3f Time: %.5f' %
              (str(epoch).zfill(3), running_loss / batches_per_dataset,
               euc.mean(), stop - start))
        return running_loss / batches_per_dataset

    def analyze_shape_loss(self, labels, pi, sigma, mu):

        # draw samples from MoG defined by parameters
        output = mdn.sample(pi, sigma, mu)

        # expand labels and output back to full size
        output = expand_labels(
            self.trainloader.dataset.full_labels[:len(labels)], output,
            self.trainloader.dataset.keep_columns,
            self.trainloader.dataset.one_columns)

        labels = expand_labels(
            self.trainloader.dataset.full_labels[:len(labels)], labels,
            self.trainloader.dataset.keep_columns,
            self.trainloader.dataset.one_columns)

        # interpret and convert to real.
        ground_truth_params = convert_dict_to_real(
            interpret_labels(labels, self.ndof),
            self.trainloader.dataset.bounds, self.ndof)
        param_dict = convert_dict_to_real(interpret_labels(output, self.ndof),
                                          self.trainloader.dataset.bounds,
                                          self.ndof)
        real_axis = ground_truth_params['axis']
        real_net_axis = param_dict['axis']

        euc = euclidean_distance(real_axis[:, :, :3],
                                 real_net_axis[:, :, :3]) * 100.0
        return euc

    def plot_losses(self):
        os.makedirs("plots/" + self.name, exist_ok=True)
        x = np.arange(len(self.losses))
        tx = np.arange(0, len(self.losses), self.test_freq)
        plt.plot(x, np.array(self.losses), color='b', label='train')
        plt.plot(tx, np.array(self.tlosses), color='r', label='test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('plots/' + self.name + '/curve.png')
        plt.close()
        np.save('plots/' + self.name + '/losses.npy', np.array(self.losses))
        np.save('plots/' + self.name + '/tlosses.npy', np.array(self.tlosses))
