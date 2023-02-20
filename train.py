from torch.optim import lr_scheduler
import torch
import os
import numpy as np
import argparse

from torch.utils.data import DataLoader
from Net import AEMP, AEMPLoss
from dataset import HyperSpecData

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.002, metavar='M',
                        help='learning rate (default: 0.002)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='epochs')
parser.add_argument('--exp_name', type=str, default='indian',
                        help='exp name')
parser.add_argument('--pca', type=int, default=16,
                        help='pca')
parser.add_argument('--patch_size', type=int, default=45,
                        help='patch')
parser.add_argument('--CUDA', type=str, default='0',
                        help='CUDA')
  
args = parser.parse_args()

cuda = args.CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
lr = args.lr
epochs = args.epochs
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA
test_step = 1
pca = args.pca
patch_size = args.patch_size 
save_epoch = 1

if not os.path.exists(args.exp_name):
    os.makedirs(args.exp_name)
    os.makedirs(os.path.join(args.exp_name,'model'))

if args.exp_name == 'indian':
    data_path = './data/Indian_pines_corrected.mat'
    label_path = './data/Indian_pines_gt.mat'
    data_key = 'indian_pines_corrected'
    label_key = 'indian_pines_gt'
elif args.exp_name == 'salinas':
    data_path = './data/Salinas_corrected.mat'
    label_path = './data/Salinas_gt.mat'
    data_key = 'salinas_corrected'
    label_key = 'salinas_gt'
elif args.exp_name == 'paviau':
    data_path = './data/PaviaU_corrected.mat'
    label_path = './data/PaviaU_gt.mat'
    data_key = 'paviaU'
    label_key = 'paviaU_gt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
dataset = HyperSpecData(data_path=data_path,
                        label_path=label_path,
                        data_key=data_key,
                        label_key=label_key,
                        train_ratio=0.1, pca=pca, patch_size=patch_size)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

net = AEMP(dataset.catalog,pca,(patch_size-3)//2+1)
celoss = AEMPLoss() 
net = net.to(device)
celoss = celoss.to(device) 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                            lr=lr, betas=(0.9, 0.999), eps=1e-8)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-7, last_epoch=-1)

from sklearn.metrics import cohen_kappa_score
def test(net, dataset, epoch, i):
    data, label, mask = dataset.get_eval()
    data = data.to(device)
    label = label.to(device)
    mask = mask.to(device)

    accs = []
    with torch.no_grad():
        net.eval()
        logist = net(data)
        pred = torch.argmax(logist,1)
        acc = torch.sum((pred == label).float()*mask) / torch.sum(mask)

        np_pred = pred.reshape(-1).detach().cpu().numpy() #batch
        np_label = label.reshape(-1).detach().cpu().numpy() #batch
        np_mask = mask.reshape(-1).detach().cpu().numpy() #batch
        np_pred_test = np_pred[np_mask!=0]
        np_label_test = np_label[np_mask!=0]
        
        for c in range(dataset.catalog):
            accs.append(np.sum((np_pred_test == c) * (np_label_test == c))/np.sum(np_label_test == c))

        kappa = cohen_kappa_score(np_label_test, np_pred_test)
        AA = np.mean(accs)
        print("epoch:{}, iter:{}, OA,{:.4f}, AA,{:.4f}, kappa,{:.4f}, acc_per,{}".format(epoch, i, acc, AA, kappa, [np.round(a, 4) for a in accs]))

    return np.array([acc.cpu().numpy(), AA, kappa])

if __name__ == '__main__':
    for epoch in range(args.epochs):
        for i, (data, label, mask) in enumerate(train_loader):
            net.train()
            data = data.to(device)
            label = label.to(device)
            mask = mask.to(device)

            logist = net(data)
            loss = celoss(logist,label,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = test(net, dataset, epoch, i)
            scheduler.step()

        if (epoch+1) % save_epoch == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            model_path = os.path.join(
                args.exp_name, 'model', 'checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
   
