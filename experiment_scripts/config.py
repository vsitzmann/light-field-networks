import os

mode = 'openmind'

if mode == 'satori':
    logging_root = '/nobackup/users/sitzmann/logs/light_fields'
    results_root = '/nobackup/users/sitzmann/results/light_fields'
    os.environ["TORCH_HOME"] = '/nobackup/users/sitzmann/'
elif mode == 'openmind':
    logging_root = '/om2/user/sitzmann/logs/light_fields'
    results_root = '/om2/user/sitzmann/results/light_fields'
    figures_root = '/om2/user/sitzmann/results/light_fields/figures'
    data_root = '/om2/user/sitzmann/'
    os.environ["TORCH_HOME"] = '/om2/user/sitzmann/'
elif mode == 'local':
    logging_root = '/home/sitzmann/test'
    results_root = '/home/sitzmann/test'
    figures_root = '/home/sitzmann/test'
    data_root = '/home/sitzmann/test'
    os.environ["TORCH_HOME"] = '/home/sitzmann/test'
