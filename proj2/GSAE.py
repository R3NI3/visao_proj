import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np

has_cuda = torch.cuda.is_available()

batch_size = 64
log_interval = 20
epochs = 100
path_resume = './save_model/trained_gsae.pth.tar'
directory = os.path.dirname(path_resume)
if not os.path.exists(directory):
    os.makedirs(directory)

kwargs = {'num_workers': 1, 'pin_memory': True} if has_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

class GSAE(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(GSAE, self).__init__()
        self.lin_encoder = nn.Linear(feature_size, hidden_size)
        self.lin_decoder = nn.Linear(hidden_size, feature_size)
        self.feature_size = feature_size
        self.hidden_size = hidden_size

    def encode(self,input):
        # encoder
        x = self.lin_encoder(input)
        x = F.relu(x)
        return x

    def decode(self,input):
        # decoder
        x = self.lin_decoder(input)
        x = F.sigmoid(x)
        return x

    def forward(self, input):
        x = input.view(-1, self.feature_size)
        x = self.encode(input)
        # sparsity penalty
        #x = L1Penalty.apply(x, self.l1weight)
        return self.decode(x).view_as(input)

model = GSAE(784, 500)
if has_cuda:
    model.cuda()

def calculate_l21_norm(X):
    norm1 = X * X
    return (np.sqrt(norm1.sum(1))).sum()

def loss_function(recon_x, x, target):
    x = x.view(-1, 784)
    lable_set = set(target)
    data = zip(x,target)
    reg_term = 0
    for lable in lable_set:
        Xc_list = [sample[0] for sample in data if sample[1] == lable]
        if Xc_list:
            X_c = torch.stack(Xc_list)
            reg_term += calculate_l21_norm(model.encode(X_c))
    mse = nn.MSELoss()
    return mse(recon_x, x) + reg_term

criterion = loss_function
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data)
        if has_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        data = data.view(-1,784)
        recon_batch = model(data)
        loss = loss_function(recon_batch, data, target)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, target) in enumerate(test_loader):
        if has_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch = model(data)
        test_loss += loss_function(recon_batch, data, target).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(batch_size, 1, 28, 28)[:n]])
            directory = os.path.dirname('./results_gsae/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_image(comparison.data.cpu(),
                 'results_gsae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def main():
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        sample = Variable(torch.randn(104, 500))
        if has_cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(104, 1, 28, 28),
                    'results_gsae/sample_' + str(epoch) + '.png')

    torch.save({
        'epoch': epochs + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}, path_resume)


if __name__ == "__main__":
    main()
