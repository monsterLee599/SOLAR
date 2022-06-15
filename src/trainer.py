#from resnet import resnet50
from utils import *
import torch.nn as nn
import torch
import os
#from torch.utils.tensorboard import SummaryWriter
#from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")
device = 'cpu'

def train_epoch(data_loader,model,optimizer,loss_function):
    model.train()
    batch_idx = -1
    for data in data_loader:
        batch_idx = batch_idx + 1
        data, target = data['feature'].to(device), data['id'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(data, output, target, model.fc.weight)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch:  [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))



def test(data_loader,model,loss_function, measure):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        real_id = torch.tensor([])
        pred_id = torch.tensor([])
        for data in data_loader:
            data, target = data['feature'].to(device), data['id'].to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_function(data, output, target, model.fc.weight).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            #print(output,pred)
            correct += pred.eq(target.view_as(pred).long()).sum().item()
            pred_id = torch.cat((pred_id.long(), pred.long()))
            real_id = torch.cat((real_id.long(), target.long()))
            measure.update(pred_id,real_id)

        test_loss /= len(data_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Recall:{}, f_measure: {}\n'
              .format(test_loss, correct, len(data_loader.dataset),
                      100. * correct / len(data_loader.dataset),measure.recall(),measure.f_measure()))

def train(train_data_loader, validate_data_loader, model, loss_function, optimizer, epoch_num=20):
    measure = metric()
    for i in range(epoch_num):
        measure.clear()
        train_epoch(train_data_loader, model, optimizer, loss_function)
        # test(validate_data_loader, model, loss_function, measure)