import torch,torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
def prep_dataset(opt):
    assert opt.dataset in ['mnist','cifar'] , 'Dataset can only be mnist or cifar10'
    train_dataset,test_dataset=None,None
    if opt.dataset=='mnist':
        transform = transforms.Compose([
        transforms.Resize(opt.ImageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5),std=(0.5))])

        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
    elif opt.dataset=='cifar':
        transform_train = transforms.Compose([
            transforms.Resize(opt.ImageSize),
            transforms.RandomCrop(opt.ImageSize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(opt.ImageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root='./cifar_data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./cifar_data/', train=False, transform=transform_test, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    concat_dataset = ConcatDataset([train_dataset, test_dataset])
    return concat_dataset,train_loader,test_loader
    

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])