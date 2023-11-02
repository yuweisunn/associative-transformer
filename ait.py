# Associative Transformer Is A Sparse Representation Learner
# See https://arxiv.org/abs/2309.12862
#
# Author: Yuwei Sun



import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
import random
import wandb
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader
import time, os
from hflayers import Hopfield, HopfieldLayer, HopfieldPooling
from hn_task import *
from MHA import *
from torch_multi_head_attention import MultiHeadAttention as mha
from triangle import *


             
class MemoryUpdate(nn.Module):
    def __init__(self, hidden_dim, l):
        super(MemoryUpdate, self).__init__()
        self.mha = MultiHeadAttention(memory_head, hidden_dim, memory_dim, memory_dim)
        self.memory = torch.nn.Parameter(torch.randn(pattern_size, memory_dim), requires_grad=False)
        self.register_buffer('beta', torch.tensor(0.9))
        self.l = l

    def forward(self, mem, moe):
        out, gates, loss = self.mha(mem.unsqueeze(0), moe.unsqueeze(0), moe.unsqueeze(0), bottleneck)
        
        momentum = self.beta
        memory = self.memory.data
        x = out.squeeze(0)
        memory.mul_(momentum)
        memory.add_(torch.mul(x.data, 1-momentum))
        w_norm = memory.pow(2).sum(1, keepdim=True).pow(0.5)
        self.memory.data = memory.div(w_norm)
        
        """
        # plot attention maps
        if iteration == 0 and torch.cuda.current_device() == 0 and (epoch+1)%100 == 0:
            np.save(f"attn_{dataset}_{epoch}_{self.l}.npy", gates.cpu().detach().numpy()) 
        """

        return  self.memory.data, loss


class ModularNN(nn.Module):
    def __init__(self, beta, l):
        super(ModularNN, self).__init__()
    
        self.memoryupdate = MemoryUpdate(hidden_dim, l)
        self.l = l
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.fc3 = nn.Linear(memory_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

        if HN:
            if not static:
                self.hopfield = Hopfield(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_heads=8,
                        update_steps_max=3,
                        scaling=beta,
                        dropout=0.5)
            else:
                self.hopfield = Hopfield(
                        scaling=beta,
                        state_pattern_as_static=True,
                        stored_pattern_as_static=True,
                        pattern_projection_as_static=True,
                        normalize_stored_pattern=False,
                        normalize_stored_pattern_affine=False,
                        normalize_state_pattern=False,
                        normalize_state_pattern_affine=False,
                        normalize_pattern_projection=False,
                        normalize_pattern_projection_affine=False,
                        disable_out_projection=True)
        else:
            self.mha_hn = mha(in_features=256, head_num=4)
    

    def forward(self, input_data):
        # FF
        aux_loss = torch.zeros(1).cuda()
        memory = self.memoryupdate.memory.data
        input_data = input_data.reshape(-1,hidden_dim)
        out = self.fc1(input_data)
        out = self.gelu(out)
        out_moe = self.norm(self.fc2(out))
        
        #Update memory
        memory, aux_loss = self.memoryupdate(memory, out_moe)
        out_moe = out_moe.reshape(-1,dim,hidden_dim)
        memory = self.fc3(memory)
        
        #MHA or Hopfield
        if not HN:
            out_hf = self.mha_hn(out_moe.reshape(-1, hidden_dim).unsqueeze(0), memory.unsqueeze(dim=0), memory.unsqueeze(dim=0))
            out_hf = out_hf.reshape(-1,dim,hidden_dim)
        else:
            out_hf = self.hopfield((memory.unsqueeze(dim=0).expand(out_moe.shape[0],*(pattern_size,hidden_dim)), out_moe, memory.unsqueeze(dim=0).expand(out_moe.shape[0],*(pattern_size,hidden_dim))))

        """
        #compute Hopfield energy
        if iteration == 5 and torch.cuda.current_device() == 0:
            energy_noisy = get_energy(out_moe[0].unsqueeze(dim=0).unsqueeze(dim=0),memory.unsqueeze(dim=0), beta)
            energy_retrieved = get_energy(out_hf[0].unsqueeze(dim=0),memory.unsqueeze(dim=0), beta)
            print(f'Energy of noisy pattern: {energy_noisy.cpu().detach().item()}, Energy of retrieved pattern (lower as expected): {energy_retrieved.cpu().detach().item()}')
        """

        #skip connection
        out = out_hf + out_moe

        return out, aux_loss
    


#settings
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='cifar10 or imagenet (default: cifar10)')
parser.add_argument('--mode', type=str, default="train",
                    help='train or eval (default: train)')
parser.add_argument('--model_size', type=str, default="small",
                    help='small or base or large (default: small)')
parser.add_argument('--epochs', type=int, default=100,
                    help='epochs (default: 100)')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size (default: 512)')
parser.add_argument('--patch_size', type=int, default=4,
                    help='patch size (default: 4)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--retrain', action='store_true',
                        help='retrain (default: False)')
parser.add_argument('--aux', action='store_false',
                        help='aux loss (default: True)')
parser.add_argument('--sparse', action='store_false',
                        help='sparse (default: True)')
parser.add_argument('--hf', action='store_false',
                        help='hopfield layer (default: True)')
parser.add_argument('--beta', type=float, default=1,    
                    help='beta (default: 1)')
parser.add_argument('--pattern_size', type=int, default=32,
                    help='pattern size (default: 32)')
parser.add_argument('--bottleneck', type=int, default=512,  
                    help='bottleneck (default: 512)')
parser.add_argument('--static', action='store_false',
                        help='static (default: True)')
parser.add_argument('--seed', type=int, default=0,
                    help='seed (default: 0)')
parser.add_argument('--memory_head', type=int, default=8,
                    help='memory head (default: 8)')
parser.add_argument('--warmup_t', type=int, default=5,
                    help='warmup time (default: 5)')
parser.add_argument('--memory_dim', type=int, default=32,
                    help='memory dim (default: 32)')
parser.add_argument('--wandb', action='store_true',
                        help='retrain (default: False)')
args = parser.parse_args()


dataset = args.dataset
mode = args.mode 
model_size = args.model_size 
epochs = args.epochs 
batch_size = args.batch_size
lr = args.lr
retrain = args.retrain
aux = args.aux
sparse = args.sparse
hf = args.hf
beta = args.beta 
pattern_size = args.pattern_size
static = args.static
bottleneck = args.bottleneck
patch_size = args.patch_size
seed = args.seed
memory_head = args.memory_head
memory_dim = args.memory_dim
warmup_t = args.warmup_t

#use self-attention
SA = True
#use Hopfield otherwise MHA
HN = True

# Set the random seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

#set wandb
if args.wandb:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Associative-Transformer",
        # Track hyperparameters and run metadata
        config={
            "dataset": dataset,
            "mode": mode,
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "retrain": retrain,
            "aux": aux,
            "sparse": sparse,
            "hf": hf,
            "beta": beta,
            "pattern_size": pattern_size,
            "static": static,
            "bottleneck": bottleneck,
            "patch_size": patch_size,
            "seed": seed,
            "memory_head": memory_head,
            "memory_dim": memory_dim
        })


# vit paramereters
if model_size == 'small':
    hidden_dim = 768
    mlp_dim = 3072 
    n_layer = 2
    heads = 12

elif model_size == 'base':
    hidden_dim = 768 
    mlp_dim = 3072
    n_layer = 12
    heads = 12 

elif model_size == 'coordination':
    hidden_dim = 256
    mlp_dim = 512
    n_layer = 1
    heads = 4 

elif model_size == 'large':
    hidden_dim = 1024 
    mlp_dim = 4096
    n_layer = 24 
    heads = 16



# datasets
if dataset == "cifar10":
    image_size = 32
    num_classes = 10
    dim = int(image_size/patch_size)**2
    # patch size 4
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)

elif dataset == "cifar100":
    image_size = 32
    num_classes = 100
    dim = int(image_size/patch_size)**2
    # patch size 4

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)

elif dataset == 'triangle':
    image_size = 64
    num_classes = 2
    dim = int(image_size/patch_size)**2
    # patch size 32

    trainset = TriangleDataset(num_examples = 50000)
    testset = TriangleDataset(num_examples = 10000)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    
elif dataset == "clevr":
    image_size = 75
    num_classes = 10
    dim = int(image_size/patch_size)**2+1
    # patch size 15/5

    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    ternary_train = []
    ternary_test = []
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []

    for img, ternary, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        img = torch.tensor(img).float()
        for qst, ans in zip(ternary[0], ternary[1]):
            qst= torch.tensor(qst).float()
            ans= torch.tensor(ans)
            ternary_train.append((img,qst,ans))
        for qst,ans in zip(relations[0], relations[1]):
            qst= torch.tensor(qst).float()
            ans= torch.tensor(ans)
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            qst= torch.tensor(qst).float()
            ans= torch.tensor(ans)
            norel_train.append((img,qst,ans))

    for img, ternary, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        img = torch.tensor(img).float()
        for qst, ans in zip(ternary[0], ternary[1]):
            qst= torch.tensor(qst).float()
            ans= torch.tensor(ans)
            ternary_test.append((img, qst, ans))
        for qst,ans in zip(relations[0], relations[1]):
            qst= torch.tensor(qst).float()
            ans= torch.tensor(ans)
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            qst= torch.tensor(qst).float()
            ans= torch.tensor(ans)
            norel_test.append((img,qst,ans))
    
    trainloader = torch.utils.data.DataLoader(CustomDataset(rel_train), batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(CustomDataset(rel_test), batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    



# initialize models

# AiT
if sparse:
    if dataset == 'clevr':
        from vit_hn_clevr import ViT 
    else:
        from vit_hn import ViT
    net = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = hidden_dim,
        depth = n_layer,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = 0.,
        emb_dropout = 0.,
        mode = mode,
        sparse = sparse
    ).cuda()

    for i in range(n_layer):
        net.transformer.layers[i][1].fn = ModularNN(beta,i).cuda()
        if not SA:                
            net.transformer.layers[i][0].fn = IdentityLayer().cuda() 

# ViT   
else:
    from vit_org import ViT
    net = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = hidden_dim,
        depth = n_layer,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = 0.,
        emb_dropout = 0.,
        mode = mode,
        sparse = sparse
    ).cuda()
        
# total model parameters
total_params = sum(p.numel() for p in net.parameters())
print(f'number of parameters - {total_params}')
if args.wandb:
    wandb.log({"number of parameters": total_params})


# training
if mode == "train":
    criterion = nn.CrossEntropyLoss()

    if retrain:
        net.load_state_dict(torch.load(f'ait_{dataset}_{model_size}_ait{sparse}_{epochs}.pth'))
        epochs = 100
        lr = 1e-6
        optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9,0.999], weight_decay=0.01) # no weight decay in case of Adam
        optimizer.load_state_dict(torch.load(f'opt_{dataset}_{model_size}_ait{sparse}_{epochs}.pth'))
    else:
        optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9,0.999], weight_decay=0.01) 
        scheduler = CosineLRScheduler(optimizer, t_initial=epochs, warmup_t=warmup_t, warmup_lr_init=1e-5, lr_min=1e-6, warmup_prefix=True) 
 
    # multi-GPUs/data parallel
    if dataset != "clevr" and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)


    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        if not retrain:
            # Update the learning rate scheduler
            scheduler.step(epoch)

        progress_bar = tqdm(total=len(trainloader))
        for iteration, data in enumerate(trainloader, 0):
            #skip any broken batch
            if data[0].shape[0] < batch_size:
                continue

            # get the inputs; data is a list of [inputs, labels]
            if dataset == 'clevr':
                inputs, qsts, labels = data
                qsts = qsts.cuda()
            else:
                inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            # forward + backward + optimize 
            if dataset == 'clevr':
                outputs, aux_loss = net(inputs, qsts) 
            else:
                outputs, aux_loss = net(inputs)

            _, predictions = torch.max(outputs, 1)
            # training acc
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            loss = criterion(outputs, labels)
            if sparse and aux:
                loss += torch.mean(aux_loss)

            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
            progress_bar.update(1)
        

        progress_bar.close()
        lr_step = optimizer.param_groups[0]['lr']
        train_acc = 100*correct/total
        


        # evaluate every epoch
        test_loss = 0.0
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(testloader):
                #skip any broken batch
                if data[0].shape[0] < batch_size:
                    continue

                if dataset == 'clevr':
                    inputs, qsts, labels = data
                    qsts = qsts.cuda()
                else:
                    inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                if dataset == 'clevr':
                    outputs, aux_loss = net(inputs, qsts) 
                else:
                    outputs, aux_loss = net(inputs)
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                loss = criterion(outputs, labels)
                if sparse and aux:
                    loss += torch.mean(aux_loss)
                test_loss += loss.cpu().item()

           
            test_acc = 100*correct/total
            print(f'[{epoch + 1}] loss: {running_loss:.3f} training acc: {train_acc:.3f} test acc: {test_acc:.3f} lr: {lr_step:.6f}')
            if args.wandb:
                wandb.log({"test accuracy": test_acc, "loss": running_loss})


    print('Finished Training')
    print("--- %s seconds ---" % (time.time() - start_time))
    if args.wandb:
        wandb.log({"Training time": (time.time() - start_time)})
        wandb.finish()
    # save the trained model weights
    torch.save(net.module.state_dict(), f'ait_{dataset}_{model_size}_ait{sparse}_{epochs}.pth')
    torch.save(optimizer.state_dict(), f'opt_{dataset}_{model_size}_ait{sparse}_{epochs}.pth')




    