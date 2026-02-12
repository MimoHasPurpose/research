from model.mask_utils import *


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix前缀 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # 引用时：model = load_model(model, cfg['model_weights'], device)
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
    else:
        device = torch.cuda.current_device()  # gpu
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test_batch_full(img1, img2, img_gt, pre_model, device, cfg):
    patch_size = cfg['patch_size']
    # 模型加载
    pre_model = load_model(pre_model, cfg['model_weights'], device)
    pre_model.eval()

    img1, img2 = img1.to(device), img2.to(device)
    C, H, W = img1.shape

    cut = PatchCut_overlap(H, W, patch_size)
    reduction = PatchReduction_overlap(H, W, patch_size)
    img1_patches = cut(img1)
    img2_patches = cut(img2)

    with torch.no_grad():
        img1_ = pre_model.init(img1_patches)
        img2_ = pre_model.init(img2_patches)
        x_d1, x_d2, x_d3, x_d4 = pre_model.encoder(img2_)
        xm_d1, xm_d2, xm_d3, xm_d4 = pre_model.encoder(img1_)
        d1, d2, d3, d4 = x_d1 - xm_d1, x_d2 - xm_d2, x_d3 - xm_d3, x_d4 - xm_d4
        x_rec = pre_model.decoder_d(d1, d2, d3, d4)
        x_rec = pre_model.mlp_d(x_rec)
        prediction = pre_model.fc(x_rec)

    prediction_full = reduction(prediction)
    predict_label = prediction_full.cpu().argmax(dim=0, keepdim=True).squeeze(0)  # torch.Size([450, 140])

    return predict_label
