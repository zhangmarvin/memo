import argparse

from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='imageneta')
parser.add_argument('--resume', default='rn50')
args = parser.parse_args()
experiment = args.experiment
resume = args.resume

dataroot = '/path/to/imagenet/datasets/'  # EDIT THIS

if experiment == 'imagenetc':
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                   'snow', 'frost', 'fog', 'brightness',
                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    levels = [1, 2, 3, 4, 5]
elif experiment == 'imagenetr':
    corruptions = ['rendition']
    levels = [0]
elif experiment == 'imageneta':
    corruptions = ['adversarial']
    levels = [0]

if resume in ('rvt', 'rn101', 'rn101_wsl'):
    model_tag = '--use_rvt' if resume == 'rvt' else '--use_resnext'
    optimizer = 'adamw'
    lr = 0.00001
    weight_decay = 0.01
else:
    model_tag = ''
    optimizer = 'sgd'
    lr = 0.00025
    weight_decay = 0.0

for corruption in corruptions:
    for level in levels:
        print(corruption, 'level', level)
        call(' '.join(['python', 'test_calls/test_initial.py',
                       f'--dataroot {dataroot}',
                       model_tag,
                       f'--level {level}',
                       f'--corruption {corruption}',
                       f'--resume results/imagenet_{resume}/']),
             shell=True)

        call(' '.join(['python', 'test_calls/test_adapt.py',
                       f'--dataroot {dataroot}',
                       model_tag,
                       f'--level {level}',
                       f'--corruption {corruption}',
                       f'--resume results/imagenet_{resume}/',
                       f'--optimizer {optimizer}',
                       f'--lr {lr}',
                       f'--weight_decay {weight_decay}']),
             shell=True)
