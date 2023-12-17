hparam = {
    # model config
    'S': 7,
    'B': 2,
    'C': 20,
    # training config
    'lr': 2e-5,
    'device': 'cuda',
    'batch_size': 16,
    'weight_decay': 0,
    'num_epochs': 100,
    # loss config
    'lambda_coord': 5,
    'lambda_noobj': 0.5,
}