def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},

        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
        }


