
sweep_alg_hparams = {
    'SHOT':
        {'pre_learning_rate': {'values': [3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'learning_rate': {'values': [5e-3, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'ent_loss_wt': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
         'im': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
         'target_cls_wt': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
         },
    'AaD':
        {'pre_learning_rate': {'values': [3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'learning_rate': {'values': [5e-3, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'beta': {'values': [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
         },
    'NRC':
        {'pre_learning_rate': {'values': [3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'learning_rate': {'values': [5e-3, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         },
    'MAPU':
        {
          'pre_learning_rate': {'values': [3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'learning_rate': {'values': [5e-3, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'ent_loss_wt': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
         'im': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
         'target_cls_wt': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
         'TOV_wt': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

        },
    'AaD_MAPU':
        {
        'pre_learning_rate': {'values': [3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'learning_rate': {'values': [5e-3, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'beta': {'values': [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
         'tov_wt': {'distribution': 'uniform', 'min': 0.001, 'max': 1}
         },
    'NRC_MAPU':
        {
        'pre_learning_rate': {'values': [3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
        'learning_rate': {'values': [5e-3, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]},
         'tov_wt': {'distribution': 'uniform', 'min': 0.001, 'max': 1}
         },

}
