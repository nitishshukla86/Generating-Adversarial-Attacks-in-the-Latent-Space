import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import copy
import time
from tqdm import tqdm
import torch.nn.functional as F  
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, ConcatDataset
from utils import prep_dataset,reset_weights
from models import Generator,Discriminator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    
    
    

def train_discriminator(real_images,labels,target):
    opt_d.zero_grad()
    real_preds = D(real_images)
    real_targets = labels
    real_loss = F.cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    # Generate fake images
    
    fake_images = G(real_images)

    # Pass fake images through discriminator
    fake_targets = torch.tensor([target]*100).to(device)
    fake_preds = D(fake_images)
    fake_loss = F.cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(real_imgs,labels,target):
    opt_g.zero_grad()
    
    # Generate fake images
    fake_images = G(real_imgs)
    
    # Try to fool the discriminator
    preds = D(fake_images)
    targets = torch.tensor([target]*100).to(device)
    loss = 0.5*F.cross_entropy(preds, targets)  +0.5*F.l1_loss(real_imgs,fake_images)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()


def fit(epochs, lr, target,trainloader,start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    
    
    for epoch in range(epochs):
        for i,(real_images, labels) in enumerate(tqdm(trainloader)):
            G.train(),D.train()
            real_images=real_images.to(device)
            labels=labels.to(device)
            
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, labels,target)
            # Train generator
            if i%opt.n_critic==0:
                loss_g = train_generator(real_images, labels,target)
        scheduler_d.step()
        scheduler_g.step()
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        
        with torch.no_grad():
            G.eval(),D.eval()
            imgs,labels=next(iter(testloader))
            imgs,labels=imgs.to(device),labels.to(device)


            num_correct=(torch.argmax(D(imgs),dim=1)==labels).sum().item()
            num_eq_to_target=(torch.argmax(D(G(imgs)),dim=1)==target).sum().item()
            acc_noisy_im=(torch.argmax(D(G(imgs)),dim=1)==labels).sum().item()
            recon=torch.abs(imgs-G(imgs)).sum().item()

            print(f'Epoch: {epoch+1}, Accuracy clean: {num_correct}, #Target {num_eq_to_target}, #Accuracy generated :{acc_noisy_im} ,recon :{recon}')

    return losses_g, losses_d, real_scores, fake_scores

def test(loader,target,ds):
    num_correct=num_eq_to_target=acc_noisy_im=recon=tot=0
    with torch.no_grad():
        G.eval(),D.eval()
        for imgs,labels in loader:
            imgs,labels=imgs.to(device),labels.to(device)
            num_correct+=(torch.argmax(D(imgs),dim=1)==labels).sum().item()
            num_eq_to_target+=(torch.argmax(D(G(imgs)),dim=1)==target).sum().item()
            acc_noisy_im+=(torch.argmax(D(G(imgs)),dim=1)==labels).sum().item()
            recon=torch.abs(imgs-G(imgs)).sum().item()
            tot+=len(labels)
    torch.save(G.state_dict(),f'./saved_models/{ds}/G_{target}_fold{fold}.pt')
    torch.save(D.state_dict(),f'./saved_models/{ds}/D_{target}_fold{fold}.pt')
    print(f' Accuracy clean: {num_correct/tot*100:.2f}, #Target {num_eq_to_target/tot*100:.2f}, #Accuracy generated :{acc_noisy_im/tot*100:.2f} ,recon :{recon:.2f}')




if __name__=='__main__':
    dataset_choices = ['cifar10','mnist']
    arch_choices = ['resnet18']
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20,help='training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=100,help='batch size for standard (regular) training')
    parser.add_argument('--lr', type=float, default=0.002,help='learning rate (default: 0.002)')
    parser.add_argument('--beta1', type=float, default=0.5,help='beta1')
    parser.add_argument('--gamma', type=float, default=0.9,help='gamma')
    parser.add_argument('--beta2', type=float, default=0.999,help='beta2')
    parser.add_argument('--dataset', type=str, default='mnist',help='dataset to use', choices=dataset_choices)
    parser.add_argument('--arch', type=str, default='resnet18',help='architecture to use', choices=arch_choices)
    parser.add_argument('--n_critic', type=int, default=50,help='number of critics')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=1,help='number of channels')
    parser.add_argument('--latent_dim', type=int, default=100,help='latent dimension')
    parser.add_argument('--k_folds', type=int, default=5,help='number of folds')
    parser.add_argument('--t', type=int, default=0,help='target class')
    parser.add_argument('--ImageSize', type=int, default=64,help='Image size')
    opt = parser.parse_args()
    if opt.dataset=='cifar10':
        opt.nc=3
    

    full_dataset,_,_ =prep_dataset(opt)
    
    kfold = KFold(n_splits=opt.k_folds, shuffle=True)
    
    G_list=[Generator(opt.nc,opt.ngf, opt.ndf, opt.latent_dim).to(device) for _ in range(opt.k_folds)]
    D_list=[Discriminator(opt.nc).to(device) for _ in range(opt.k_folds)]
    
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        G=G_list[fold]
        D=D_list[fold]
        opt_g = torch.optim.Adam(G.parameters(), lr=opt.lr,betas=(opt.beta1, opt.beta2))
        opt_d = torch.optim.Adam(D.parameters(), lr=opt.lr,betas=(opt.beta1, opt.beta2))  
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=opt.gamma)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=opt.gamma)
        reset_weights(G)
        reset_weights(D)

        print(f'______training fold {fold} : target {opt.t}______')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                          full_dataset, 
                          batch_size=opt.batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          full_dataset,
                          batch_size=opt.batch_size, sampler=test_subsampler)
        fit(opt.epochs, opt.lr, opt.t,trainloader)

        test(testloader,opt.t,opt.dataset)
        print('\n\n')




