import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter

#-------------------------------------------------------------------------------
def predict_epoch(model, predict_loader, device):
    prediction = np.array([])
    loop = tqdm(enumerate(predict_loader), total = len(predict_loader))
    for batch_idx, (batch_data_t1, batch_data_t2, batch_label) in loop:
        batch_data_t1 = batch_data_t1.to(device)
        batch_data_t2 = batch_data_t2.to(device)
        batch_label = batch_label.to(device).long()

        # predict the data without label
        batch_prediction = model(batch_data_t1, batch_data_t2)

        # the maximum of the possibility 
        _, pred = batch_prediction.topk(1, 1, True, True)
        pred_squeeze = pred.squeeze()
        prediction = np.append(prediction, pred_squeeze.data.cpu().numpy())

        loop.set_description(f'Test Epoch')  
    return prediction
#-------------------------------------------------------------------------------