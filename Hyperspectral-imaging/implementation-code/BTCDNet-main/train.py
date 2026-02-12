import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.metrics import accuracy

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, e, epoch, device):
    loss_show = AverageMeter()
    acc = AverageMeter()
    label = np.array([])
    prediction = np.array([])
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (batch_data_t1, batch_data_t2, batch_label) in loop:
        batch_data_t1 = batch_data_t1.to(device)
        batch_data_t2 = batch_data_t2.to(device)
        batch_label = batch_label.to(device).long()

        optimizer.zero_grad()
        batch_prediction = model(batch_data_t1, batch_data_t2)
        loss = criterion(batch_prediction, batch_label)
        loss.backward()
        optimizer.step()       

        # calculate the accuracy
        acc_batch, l, p = accuracy(batch_prediction, batch_label, topk=(1,))
        n = batch_label.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc.update(acc_batch[0].data, n)
        label = np.append(label, l.data.cpu().numpy())
        prediction = np.append(prediction, p.data.cpu().numpy())

        loop.set_description(f'Train Epoch [{e+1}/{epoch}]')
        loop.set_postfix({"train_loss":loss_show.average.item(),
                          "train_accuracy": str(round(acc.average.item(), 2)) + "%"})

    return acc.average, loss_show.average, label, prediction
#-------------------------------------------------------------------------------

# test model
def test_epoch(model, test_loader, criterion, device):
    loss_show = AverageMeter()
    acc = AverageMeter()
    label = np.array([])
    prediction = np.array([])
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    for batch_idx, (batch_data_t1, batch_data_t2, batch_label) in loop:
        batch_data_t1 = batch_data_t1.to(device)
        batch_data_t2 = batch_data_t2.to(device)
        batch_label = batch_label.to(device).long()

        batch_prediction = model(batch_data_t1, batch_data_t2)
        loss = criterion(batch_prediction, batch_label)

        # calculate the accuracy
        acc_batch, l, p = accuracy(batch_prediction, batch_label, topk=(1,))
        n = batch_label.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc.update(acc_batch[0].data, n)
        label = np.append(label, l.data.cpu().numpy())
        prediction = np.append(prediction, p.data.cpu().numpy())

        loop.set_description(f'Test Epoch')
        loop.set_postfix({"test_loss":loss_show.average.item(),
                          "test_accuracy": str(round(acc.average.item(), 2)) + "%"})
        
    return label, prediction
#-------------------------------------------------------------------------------