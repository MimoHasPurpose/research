current_dataset = 'Farmland'
current_model = '_A'
current = current_dataset + current_model

phase = ['tru', 'trl', 'test', 'gt']
train_set_num = [0.2, 0.0002]

patch_size = 32
lr = [1e-3, 1e-2]
epoch_number = 2  # epoch=20
bs_number = 128
lr_step_pre = [30, 50, 50]

data = dict(
    current_dataset=current_dataset,
    train_set_num=train_set_num,
    patch_size=patch_size,
    trl_data=dict(
        phase=phase[0]
    ),
    tru_data=dict(
        phase=phase[1]
    ),

    test_data=dict(
        phase=phase[2]
    ),
)

model = {'Farmland': [155, 450, 140], 'Hermiston': [154, 307, 241], 'Bayarea': [224, 600, 500]}

train = dict(
    optimizer_pre=dict(
        typename='SGD',
        lr=lr[0],
        momentum=0.9,
        weight_decay=1e-3
    ),
    optimizer_ft=dict(
        typename='SGD',
        lr=lr[1],
        momentum=0.9,
        weight_decay=1e-3
    ),
    train_pre_model=dict(
        patch_size=patch_size,
        gpu_train=True,
        gpu_num=1,
        workers_num=12,
        epoch=epoch_number,
        batch_size_u=bs_number,
        lr=lr[0],
        lr_adjust=True,
        lr_gamma=0.1,
        lr_step=lr_step_pre,
        save_folder='./weights/' + current_dataset + '/',
        save_name=current + '_Final',
        reuse_model=False,
        reuse_file='./weights/' + current + '_Final.pth',
    ),
)

test = dict(
    batch_size=500,
    patch_size=patch_size,
    gpu_train=True,
    gpu_num=1,
    workers_num=8,
    model_weights='./weights/' + current_dataset + '/' + current + '_Final.pth',
    save_name=current,
    save_folder='./result' + '/' + current_dataset
)
