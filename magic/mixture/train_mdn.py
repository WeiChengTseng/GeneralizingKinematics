import argparse
import torch
import numpy as np
from torchvision import transforms

from magic.noise_models import DropPixels, Noise
from magic.mixture.mixture_density_trainer import MDNTrainer
from magic.mixture.dataset import MixtureDataset
from magic.mixture.models import KinematicMDN, KinematicMDNv2, KinematicMDNv3
from magic.mixture.mixture_errors import mixture_error_table

parser = argparse.ArgumentParser(
    description="Train object learner on articulated object dataset.")
parser.add_argument('--name', type=str, help='jobname', default='test')
parser.add_argument('--n_gaussians',
                    type=int,
                    help='number of components in mixture',
                    default=20)
parser.add_argument('--train-dir',
                    type=str,
                    default='../data/524test2/microwave/')
parser.add_argument('--test-dir',
                    type=str,
                    default='../data/524test2/microwave-test')
parser.add_argument('--ntrain', type=int, default=16)
parser.add_argument('--ntest',
                    type=int,
                    default=2,
                    help='number of test samples')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help='number of iterations through data')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--test-freq',
                    type=int,
                    default=20,
                    help='frequency at which to test')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda',
                    action='store_true',
                    default=False,
                    help='use cuda')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--normalize',
                    action='store_true',
                    default=False,
                    help='scale target values to [0,1]')
parser.add_argument('--hist',
                    action='store_true',
                    default=False,
                    help='examine histogram of predictions after training')
parser.add_argument('--nwork', type=int, default=8, help='num_workers')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-5,
                    type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--ndof',
                    type=int,
                    default=1,
                    help='how many degrees of freedom in the object class?')
parser.add_argument('--obj', type=str, default='microwave')
parser.add_argument('--drop', type=float, default=0.8, help='dropout prob')
parser.add_argument('--device', type=int, default=0, help='cuda device')
args = parser.parse_args()

print(args)
print('cuda?', torch.cuda.is_available())

# make a models directory
os.makedirs('models', exist_ok=True)

# set seed for reproducibility
torch.manual_seed(args.seed)

# setup dataset, create DataLoaders
# noiser = transforms.Compose([Noise(0.0, 0.005), DropPixels(p=0.1)])
noiser = DropPixels(p=0.1)
trainset = MixtureDataset(args.ntrain,
                          args.train_dir,
                          n_dof=args.ndof,
                          normalize=True,
                          transform=noiser)

testset = MixtureDataset(args.ntest,
                         args.test_dir,
                         n_dof=args.ndof,
                         normalize=True,
                         transform=noiser,
                         bounds=trainset.bounds,
                         keep_columns=trainset.keep_columns,
                         one_columns=trainset.one_columns)

np.save('keep_columns_' + args.obj, trainset.keep_columns)
np.save('one_columns_' + args.obj, trainset.one_columns)

# DEBUG MODE ###################################################################
# train_mask = np.arange(0,160,80)
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_mask)
# test_mask = np.arange(0,10,5)
# test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_mask)
# testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
#                                         shuffle=False, num_workers=args.nwork,
#                                         pin_memory=True, sampler=test_sampler)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
#                                         shuffle=False, num_workers=args.nwork,
#                                         pin_memory=True, sampler=train_sampler)
# ##############################################################################

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch,
                                         shuffle=True,
                                         num_workers=args.nwork,
                                         pin_memory=True)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch,
                                          shuffle=True,
                                          num_workers=args.nwork,
                                          pin_memory=True)

# init model
network = KinematicMDNv3(n_gaussians=args.n_gaussians,
                         out_features=trainset.labels.shape[1],
                         p=args.drop)

# setup trainer
if torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device('cpu')

optimizer = torch.optim.Adam(network.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay)
trainer = MDNTrainer(network,
                     trainloader,
                     testloader,
                     optimizer,
                     args.epochs,
                     args.name,
                     args.test_freq,
                     device,
                     obj=args.obj,
                     ndof=args.ndof)
# train
best_model = trainer.train()

# compute error table
mixture_error_table(
    args.name,
    best_model,
    testloader,
    torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    args.n_gaussians,
    args.obj,
    args.ndof,
    normalize=True,
    sample_error=False)
