import torch
hparam = {
    # model config
    'S': 4,
    'B': 2,
    'dropout': 0.5,
    'image_size': 256,
    # training config
    'lr': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'weight_decay': 0,
    'num_epochs': 10,
    'num_worker': 0,
    'Pin_memory': True,
    'load_model': True,
    'load_model_file': 'overfit.pth.tar',
    'max_training_samples': 100,
    # loss config
    'lambda_coord': 5,
    'lambda_noobj': 0.5,
}