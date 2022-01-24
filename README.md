# MEMO: Test Time Robustness via Adaptation and Augmentation

These directories contain code for reproducing the MEMO results for the CIFAR-10
and ImageNet distribution shift test sets.

_Please note:_ this code has been modified from the version that generated the
results in the paper for the purpose of cleaning the code. Though it is likely
that the code is correct and should produce the same results, it is possible
that there may be some discrepancies that were not caught. Though minor results
differences may arise from stochasticity, please report any major differences or
bugs by submitting an issue, and we will aim to resolve these promptly.


## Setup

First, create a Anaconda environment with `requirements.txt`, e.g.,
```
conda create -n memo python=3.8 -y -q --file requirements.txt
conda activate memo
```

After doing so, you will need to `pip install tqdm`. For the robust vision
transformer models, you will also need to `pip install timm einops`.


## CIFAR-10 Experiments

The `cifar-10-exps` directory contains code for the CIFAR-10 experiments. You
can run `bash script_c10.sh` for the full set of experiments. Alternatively, you
can run `python script_test_c10.py` directly with the experiment you wish to run
(see `script_c10.sh` for more details).

For convenience, we provide the ResNet26 model that we trained in
`results/cifar10_rn26_gn/ckpt.pth`. We do not provide the datasets themselves,
though you can download the non standard test sets here:

- [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1)
- [CIFAR-10-C](https://zenodo.org/record/2535967)

After downloading and setting up the datasets, make sure to modify the
`dataroot` variable on line 8 of `script_test_c10.py`.


## ImageNet Experiments

The `imagenet-exps` directory contains code for the ImageNet experiments. You
can run `bash script_in.sh` for the full set of experiments, though this is very
slow. You can again run `python script_test_in.py` directly with the experiment
you wish to run. For the corrupted image datasets, you may wish to slightly
modify the code to only run one corruption-level pair (and then parallelize).

As an example, we provide the baseline ResNet-50 model from `torchvision` in
`results/imagenet_rn50/ckpt.pth`. Including all of the pretrained model weights
would be prohibitively large. We did not train our own models for ImageNet, and
all other models we used can be downloaded:

- [ResNet-50 w/ DeepAugment + AugMix](https://drive.google.com/file/d/1QKmc\_p6-qDkh51WvsaS9HKFv8bX5jLnP)
- [ResNet-50 w/ moment exchange + CutMix](https://drive.google.com/file/d/1cCvhQKV93pY-jj8f5jITywkB9EabiQDA)
- [RVT\*-small](https://drive.google.com/file/d/1g40huqDVthjS2H5sQV3ppcfcWEzn9ekv)
- [ResNext-101 (32x8d) w/ WSL](https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth)

We also experimented with a baseline ResNext-101 (32x8d) model which we obtained
from `torchvision`.

_Please note:_ some of these models provide the weights in slightly different
conventions, thus loading the downloaded `state_dict` may not directly work, and
the keys in the `state_dict` may need to be modified to match with the code. We
have done this modification already for the baseline ResNet-50 model, and thus
this `ckpt.pth` can be used as a template for modifying other model checkpoints.

We again do not provide the datasets themselves, though you can download the
test sets here:

- [ImageNet-C](https://zenodo.org/record/2235448)
- [ImageNet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [ImageNet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)

After downloading and setting up the datasets, again make sure to modify the
`dataroot` variable on line 8 of `script_test_in.py`.


## Paper

Please use the following citation:

```
@article{memo,
    author={Zhang, M. and Levine, S. and Finn, C.},
    title={{MEMO}: Test Time Robustness via Adaptation and Augmentation},
    article={arXiv preprint arXiv:2110.09506},
    year={2021},
}
```

The paper can be found on arXiv [here](https://arxiv.org/abs/2110.09506).


## Acknowledgments

The design of this code was adapted from the
[TTT](https://github.com/yueatsprograms/ttt_cifar_release)
[codebases](https://github.com/yueatsprograms/ttt_imagenet_release). Other parts
of the code that were adapted from third party sources are credited via comments
in the code itself.
