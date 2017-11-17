import torch
import torch.utils.data
from SAE import SAE
from fc import fc_model
from torch import nn, optim
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

has_cuda = torch.cuda.is_available()

path_resume_sae = './save_model/trained.pth.tar'
path_resume_fc = './save_model/trained_fc.pth.tar'
path_resume = './save_model/trained_model.pth.tar'
directory = os.path.dirname(path_resume)
if not os.path.exists(directory):
    os.makedirs(directory)

batch_size = 64
test_batch_size = 10000
log_interval = 100
epochs = 100

kwargs = {'num_workers': 1, 'pin_memory': True} if has_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=test_batch_size, shuffle=True, **kwargs)

class final_model(nn.Module):
    def __init__(self, feature_sz, hidden_sz, output_sz, l1weight):
        super(final_model, self).__init__()
        self.my_sae = SAE(feature_sz, hidden_sz, l1weight)
        self.my_fc = fc_model(hidden_sz, output_sz)

    def load_base_models(self, path_resume_sae, path_resume_fc):
        if os.path.isfile(path_resume_sae):
            print("=> loading checkpoint '{}'".format(path_resume_sae))
            checkpoint = torch.load(path_resume_sae)
            self.my_sae.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path_resume_sae, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path_resume_sae))

        if os.path.isfile(path_resume_fc):
            print("=> loading checkpoint '{}'".format(path_resume_fc))
            checkpoint = torch.load(path_resume_fc)
            self.my_fc.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path_resume_fc, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path_resume_fc))

    def forward(self, data):
        x = self.my_sae.encode(data)
        return self.my_fc(x)

model = final_model(784, 500, 10, 0.01)
model.load_base_models(path_resume_sae, path_resume_fc)
if has_cuda:
    model.cuda()

criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(target)
        if has_cuda:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        data = data.view(-1,784)
        y_pred = model(data)

        loss = criterion(y_pred, target)
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
            target = target.cuda()
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        y_pred = model(data)
        test_loss += criterion(y_pred, target).data[0]
        pred = y_pred.data.max(1)[1]
        d = pred.eq(target.data).cpu()
        accuracy = d.sum()/d.size()[0]
        print('====> Test Epoch:{} Accuracy {}'.format(epoch, accuracy))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def main():
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)

    torch.save({
            'epoch': epochs + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, path_resume)


if __name__ == "__main__":
    main()