import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import ConfusionMatrixDisplay
import yaml
import torch
import argparse

from models.SSTFormer import SSTFormer
from models.BTCDNet import BTCDNet
from models.BTCDNet_no import BTCDNet_no
from models.BTCDNet_noP import BTCDNet_noP
from models.BTCDNet_noT import BTCDNet_noT
from models.CSANet import CSANet
from models.BIT import BIT
from models.GETNET import GETNET
from models.ReCNN import ReCNN
from utils.dataloader import HSI_Dataset
from utils.metrics import output_metric
from utils.utils import load_hsi_mat_data
from train import train_epoch, test_epoch
from predict import predict_epoch

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config):
    print("===============================================================================")
    print("config:")
    # Extract data settings
    HSI_data_path = config['data']['HSI_data_path']
    num_samples = config['data']['num_samples']
    
    # Extract settings
    mode = config['settings']['mode']
    pretrained = config['settings']['pretrained']
    model_path = config['settings']['model_path']
    DP = config['settings']['DP']
    gpu = config['settings']['gpu']
    epoch = config['settings']['epoch']
    test_freq = config['settings']['test_freq']
    batch_size = config['settings']['batch_size']
    learning_rate = config['settings']['learning_rate']
    min_learning_rate = config['settings']['min_learning_rate']
    weight_decay = config['settings']['weight_decay']
    ignore_index = config['settings']['ignore_index']
    random_seed = config['settings']['random_seed']

    # Extract model settings
    model_type = config['model']['model_type']
    patches = config['model']['patches']
    band_patches = config['model']['band_patches']
    num_classes = config['model']['num_classes']

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # GPU configuration
    if DP and gpu:
        print(f"Using GPUs: {gpu}")
    
    # Print loaded configuration
    print("Configuration loaded successfully:")
    print(config)
    
    # Example: Initialize model, dataset, and training loop
    if mode == "train":
        print(f"Training {model_type} model...")
        # Initialize your model and training logic here
    elif mode == "test":
        print(f"Testing {model_type} model from {model_path}...")
        # Initialize your testing logic here
    else:
        raise ValueError("Invalid mode. Please choose 'train' or 'test'.")

    print("===============================================================================")
    # time setting
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cudnn.deterministic = True
    cudnn.benchmark = False
    #-------------------------------------------------------------------------------

    # make the run folder in logs
    #-------------------------------------------------------------------------------
    time_now = time.localtime()
    time_folder = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time_now)+"_"+model_type)
    os.makedirs(time_folder, exist_ok=True)

    # Initialize TensorBoard writer
    os.makedirs(os.path.join(time_folder, "SummaryWriter"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(time_folder, "SummaryWriter"))

    # Log basic parameters
    params = {
        "Model Type": model_type,
        "Epochs": epoch,
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "Weight Decay": weight_decay,
        "GPU(s)": gpu,
        "training samples": num_samples,
    }

    for key, value in params.items():
        writer.add_text(key, str(value), 0)  
    #-------------------------------------------------------------------------------

    # data loading
    #-------------------------------------------------------------------------------
    data_t1, data_t2, train_label, test_label, all_label, negative_num, positive_num = load_hsi_mat_data(HSI_data_path, num_samples)
    plt.imsave(os.path.join(time_folder, "train_label.png"), train_label, cmap='gray_r', dpi=300)
    plt.imsave(os.path.join(time_folder, "test_label.png"), test_label, cmap='gray_r', dpi=300)
    plt.imsave(os.path.join(time_folder, "data_label.png"), all_label, cmap='gray_r', dpi=300)
    train_label_image = plt.imread(os.path.join(time_folder, "train_label.png"))
    test_label_image = plt.imread(os.path.join(time_folder, "test_label.png"))
    data_label_image = plt.imread(os.path.join(time_folder, "data_label.png"))
    writer.add_image('Train Label', train_label_image, 0, dataformats='HWC')
    writer.add_image('Test Label', test_label_image, 0, dataformats='HWC')
    writer.add_image('Data Label', data_label_image, 0, dataformats='HWC')

    if data_t1.shape == data_t2.shape and len(data_t1.shape) == 3 and len(data_t2.shape) == 3:
        height, width, band = data_t1.shape
    else:
        raise ValueError("data size error!")
    #-------------------------------------------------------------------------------

    # models
    #-------------------------------------------------------------------------------
    if model_type == "SSTViT":
        if band_patches >= 3:
            model = SSTFormer(
                image_size = patches,
                near_band = band_patches,
                num_patches = band,
                num_classes = num_classes,
                dim = 32,
                depth = 2,
                heads = 4,
                dim_head = 16,
                mlp_dim = 8,
                b_dim = 512,
                b_depth = 3,
                b_heads = 8,
                b_dim_head= 32,
                b_mlp_head = 8,
                dropout = 0.2,
                emb_dropout = 0.1,
            )
        else:
            raise ValueError("band patches error!")

    elif model_type == "CSANet":
        if band_patches == 1:
            model = CSANet(
                in_chans=band, 
                num_classes=num_classes
            )
        else:
            raise ValueError("band patches error!")

    elif model_type == "BIT":
        if band_patches == 1:
            model = BIT(
                in_chans=band, 
                num_classes=num_classes
            )
        else:
            raise ValueError("band patches error!")

    elif model_type == "GETNET":
        if band_patches == 1:
            model = GETNET(
                in_chans=band, 
                num_classes=num_classes
            )
        else:
            raise ValueError("band patches error!")

    elif model_type == "ReCNN":
        if band_patches == 1:
            model = ReCNN(
                in_chans=band, 
                num_classes=num_classes
            )
        else:
            raise ValueError("band patches error!")
        
    elif model_type == "BTCDNet_no":
        if band_patches == 1:
            model = BTCDNet_no(
                    in_chans=band, 
                    num_classes=num_classes, 
                    embed_dim=[96, 192], 
                    depths=[2, 2], 
                    num_heads=[3, 6], 
                    n_iter=[1, 1], 
                    stoken_size=[2, 1], 
                    mlp_ratio=4, 
                    qkv_bias=True, 
                    qk_scale=None, 
                    drop_rate=0, 
                    attn_drop_rate=0, 
                    drop_path_rate=0.1, 
                    projection=512, 
                    freeze_bn=False, 
                    layerscale=[False, False, False, False], 
                    init_values=1e-6,
                    temperature=0.5
                )
        else:
            raise ValueError("band patches error!")

    elif model_type == "BTCDNet_noT":
        if band_patches == 1:
            model = BTCDNet_noT(
                    in_chans=band, 
                    num_classes=num_classes, 
                    embed_dim=[96, 192], 
                    depths=[2, 2], 
                    num_heads=[3, 6], 
                    n_iter=[1, 1], 
                    stoken_size=[2, 1], 
                    mlp_ratio=4, 
                    qkv_bias=True, 
                    qk_scale=None, 
                    drop_rate=0, 
                    attn_drop_rate=0, 
                    drop_path_rate=0.1, 
                    projection=512, 
                    freeze_bn=False, 
                    layerscale=[False, False, False, False], 
                    init_values=1e-6,
                    temperature=0.5
                )
        else:
            raise ValueError("band patches error!")
        
    elif model_type == "BTCDNet_noP":
        if band_patches == 1:
            model = BTCDNet_noP(
                    in_chans=band, 
                    num_classes=num_classes, 
                    embed_dim=[96, 192], 
                    depths=[2, 2], 
                    num_heads=[3, 6], 
                    n_iter=[1, 1], 
                    stoken_size=[2, 1], 
                    mlp_ratio=4, 
                    qkv_bias=True, 
                    qk_scale=None, 
                    drop_rate=0, 
                    attn_drop_rate=0, 
                    drop_path_rate=0.1, 
                    projection=512, 
                    freeze_bn=False, 
                    layerscale=[False, False, False, False], 
                    init_values=1e-6,
                    temperature=0.5
                )
        else:
            raise ValueError("band patches error!")

    elif model_type == "BTCDNet":
        if band_patches == 1:
            model = BTCDNet(
                    in_chans=band, 
                    num_classes=num_classes, 
                    embed_dim=[96, 192], 
                    depths=[2, 2], 
                    num_heads=[3, 6], 
                    n_iter=[1, 1], 
                    stoken_size=[2, 1], 
                    mlp_ratio=4, 
                    qkv_bias=True, 
                    qk_scale=None, 
                    drop_rate=0, 
                    attn_drop_rate=0, 
                    drop_path_rate=0.1, 
                    projection=512, 
                    freeze_bn=False, 
                    layerscale=[False, False, False, False], 
                    init_values=1e-6,
                    temperature=0.5
                )
        else:
            raise ValueError("band patches error!")
    #-------------------------------------------------------------------------------

    # obtain dataset and dataloader
    #-------------------------------------------------------------------------------
    train_dataset = HSI_Dataset(data_t1, data_t2, train_label, patches, band_patches, "train")
    test_dataset = HSI_Dataset(data_t1, data_t2, test_label, patches, band_patches, "train")
    all_dataset = HSI_Dataset(data_t1, data_t2, all_label, patches, band_patches, "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
    #-------------------------------------------------------------------------------

    # model settings
    #-------------------------------------------------------------------------------
    if DP:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # criterion
    # weight = torch.tensor([0, positive_num/(positive_num+negative_num), negative_num/(positive_num+negative_num)])
    # weight = torch.tensor([0, negative_num/(positive_num+negative_num), positive_num/(positive_num+negative_num)])
    # criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//25, eta_min=min_learning_rate)
    #-------------------------------------------------------------------------------

    # start training or testing
    #-------------------------------------------------------------------------------
    if mode == "train":
        # if pretrained
        if pretrained:
            model.load_state_dict(torch.load(model_path))
            print("load model path : " + model_path)
        # train
        print("===============================================================================")
        print("start training")
        tic = time.time()
        best_loss = 1e9
        for e in range(epoch): 
            model.train()
            train_acc, train_loss, label_t, prediction_t = train_epoch(model, train_loader, criterion, optimizer, e, epoch, device)
            scheduler.step()
            OA_train, AA_train, Kappa_train, CA_train, CM_train = output_metric(label_t, prediction_t, ignore_index=0) 
            print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.4f}%".format(e+1, train_loss, train_acc))
            
            # Logging to TensorBoard
            writer.add_scalar('Loss/train', train_loss, e+1)
            writer.add_scalar('Accuracy/train', train_acc, e+1)
            writer.add_scalar('AA/train', AA_train, e+1)  
            writer.add_scalar('Kappa/train', Kappa_train, e+1)  

            if train_loss < best_loss:
                torch.save(model, os.path.join(time_folder, "model_best.pt"))
                torch.save(model.state_dict(), os.path.join(time_folder, "model_state_dict_best.pth"))
                best_loss = train_loss
        
            if ((e+1) % test_freq == 0) | (e == epoch - 1):
                print("===============================================================================")
                print("start testing")      
                model.eval()
                label_v, prediction_v = test_epoch(model, test_loader, criterion, device)
                OA_test, AA_test, Kappa_test, CA_test, CM_test = output_metric(label_v, prediction_v, ignore_index=0)
                if (e != epoch -1):
                    print("Epoch: {:03d}  =>  OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(e+1, OA_test, AA_test, Kappa_test))
                
                # Logging test metrics to TensorBoard
                writer.add_scalar('Accuracy/test', OA_test, e+1)
                writer.add_scalar('AA/test', AA_test, e+1) 
                writer.add_scalar('Kappa/test', Kappa_test, e+1)  
                
                cm_display = ConfusionMatrixDisplay(CM_test)
                cm_display.plot(cmap='Blues', colorbar=False)
                plt.title('Confusion Matrix')
                plt.tight_layout()
                cm_image_path = os.path.join(time_folder, f"confusion_matrix_{e+1}.png")
                plt.savefig(cm_image_path, dpi=300)
                plt.close() 
                cm_image = plt.imread(cm_image_path)
                writer.add_image(f'Confusion Matrix/Epoch {e+1}', cm_image, e+1, dataformats='HWC')

                print("===============================================================================")

        toc = time.time()
        print("Running Time: {:.2f}".format(toc-tic))
        print("end training and testing")
        print("===============================================================================")
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA_test, AA_test, Kappa_test))
        print("CA:", end="")
        print(CA_test)
        print("Confusion Matrix:")
        print(CM_test)
        print("===============================================================================")

    elif mode == "test":
        model.load_state_dict(torch.load(model_path))
        print("load model path : " + model_path)
        print("===============================================================================")

    else:
        raise ValueError("model error!")

    if mode == "train":
        # save model and its parameters 
        torch.save(model, os.path.join(time_folder, "model_last.pt"))
        torch.save(model.state_dict(), os.path.join(time_folder, "model_state_dict_last.pth"))

    print("start predicting")
    tic = time.time()
    model.load_state_dict(torch.load(os.path.join(time_folder, "model_state_dict_best.pth")))
    model.eval()

    # output classification maps
    pre_u = predict_epoch(model, all_loader, device)
    prediction = np.zeros((height, width), dtype=float)
    for idx, (row, col) in enumerate(np.ndindex(height, width)):
        if all_label[row, col] != 0:
            prediction[row, col] = pre_u[idx]  
    plt.imsave(os.path.join(time_folder, "predict_result.png"), prediction, cmap='gray_r', dpi=300)

    # write prediction image to TensorBoard
    prediction_image = plt.imread(os.path.join(time_folder, "predict_result.png"))
    writer.add_image('Prediction Result', prediction_image, 0, dataformats='HWC')

    # close the TensorBoard writer
    writer.close()

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("end predicting")
    print("===============================================================================")

    # tensorboard --logdir=/home/ljs/BCDMNet/logs/2024-10-22-04-31-09/SummaryWriter --port=6061
    # ssh -NfL 8080:127.0.0.1:6061 ljs@172.18.206.54 -p 6522

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BTCDNet")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
