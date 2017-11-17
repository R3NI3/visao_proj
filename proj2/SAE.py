import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

batch_size = 64
log_interval = 20
epochs = 100
path_resume = './save_model/trained.pth.tar'
training = True
directory = os.path.dirname(path_resume)
if not os.path.exists(directory):
    os.makedirs(directory)

has_cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if has_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

class L1Penalty(Function):
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(self.l1weight)
        grad_input += grad_output
        return grad_input


class SAE(nn.Module):
    def __init__(self, feature_size, hidden_size, l1weight):
        super(SAE, self).__init__()
        self.lin_encoder = nn.Linear(feature_size, hidden_size)
        self.lin_decoder = nn.Linear(hidden_size, feature_size)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.l1weight = l1weight

    def encode(self,input):
        # encoder
        x = input.view(-1, self.feature_size)
        x = self.lin_encoder(x)
        x = F.relu(x)
        return x

    def decode(self,input):
        # decoder
        x = self.lin_decoder(input)
        x = F.sigmoid(x)
        return x

    def forward(self, input):
        x = self.encode(input)
        # sparsity penalty
        x = L1Penalty.apply(x, self.l1weight)
        return self.decode(x).view_as(input)

model = SAE(784, 500, 0.01)
if has_cuda:
    model.cuda()

criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1) not available yet


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if has_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        data = data.view(-1,784)
        recon_batch = model(data)
        loss = criterion(recon_batch, data)
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
    for i, (data, _) in enumerate(test_loader):
        if has_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch = model(data)
        test_loss += criterion(recon_batch, data).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(batch_size, 1, 28, 28)[:n]])
            directory = os.path.dirname('./results_sae/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_image(comparison.data.cpu(),
         'results_sae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def load_model():
    if os.path.isfile(path_resume):
        print("=> loading checkpoint '{}'".format(path_resume))
        checkpoint = torch.load(path_resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path_resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path_resume))

def main():
    if(training):
        for epoch in range(1, epochs + 1):
            train(epoch)
            test(epoch)
            sample = Variable(torch.randn(104, 500))
            if has_cuda:
                sample = sample.cuda()
            sample = model.decode(sample).cpu()
            save_image(sample.data.view(104, 1, 28, 28),
                        'results_sae/sample_' + str(epoch) + '.png')

        torch.save({
            'epoch': epochs + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, path_resume)

    else:
        load_model()
        test(0)
        #use for something

#


if __name__ == "__main__":
    main()
