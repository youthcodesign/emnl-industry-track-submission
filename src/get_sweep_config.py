import wandb

def get_base_sweep_config():
    return {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val_f1_macro',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 2e-5,
                'max': 5e-3
            },
            'batch_size': {
                'values': [8, 16, 32]
            },
            'num_epochs': {
                'values': [2, 5]
            },
            'l2_lambda': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'ewc_lambda': {
                'distribution': 'log_uniform_values',
                'min': 1e2,
                'max': 1e4
            },
            'focal_alpha': {
                'distribution': 'uniform',
                'min': 0.25,
                'max': 1.0
            },
            'focal_gamma': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 3.0
            },
            'use_focal_loss': {
                'values': [True, False]
            },
            'use_ewc': {
                'values': [False]
            },
            'dropout_rate': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'warmup_steps': {
                'values': [100, 500, 1000]
            }
        }
    }

def get_multiclass_scratch_config():
    config = get_base_sweep_config()
    # Not needed for scratch training
    config['parameters'].pop('ewc_lambda')  
    config['parameters'].pop('use_ewc')
    return config

def get_continual_learning_config():
    return get_base_sweep_config()  

def get_role_specific_config():
    return get_base_sweep_config()  

def get_role_pair_config():
    return get_base_sweep_config()  
    
# Create sweep configurations for each strategy
SWEEP_CONFIGS = {
    'multiclass_scratch': {
        'program': 'main.py',
        'project': 'bullying-classification-multiclass-scratch',
        **get_multiclass_scratch_config()
    },
    'role_pair': {
        'program': 'main.py',
        'project': 'bullying-classification-role-pair',
        **get_role_pair_config()
    }
}