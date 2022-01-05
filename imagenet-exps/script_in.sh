export PYTHONPATH=$PYTHONPATH:$(pwd)

# ImageNet-A
python script_test_in.py --experiment imageneta --resume rn50       # Baseline ResNet-50
python script_test_in.py --experiment imageneta --resume rn50_daam  # ResNet-50 + DeepAugment + AugMix
python script_test_in.py --experiment imageneta --resume rn50_moex  # ResNet-50 + moment exchange + CutMix
python script_test_in.py --experiment imageneta --resume rvt        # RVT*-small
python script_test_in.py --experiment imageneta --resume rn101      # Baseline ResNext-101
python script_test_in.py --experiment imageneta --resume rn101_wsl  # ResNext-101 + weakly supervised learning

# ImageNet-R
python script_test_in.py --experiment imagenetr --resume rn50
python script_test_in.py --experiment imagenetr --resume rn50_daam
python script_test_in.py --experiment imagenetr --resume rn50_moex
python script_test_in.py --experiment imagenetr --resume rvt

# ImageNet-C
# the following commands take much, much longer to run
# here, it is best to modify script_test_in.py to run single corruption-level pairs,
# and then parallelize the calls to script_test_in.py across multiple machines
python script_test_in.py --experiment imagenetc --resume rn50
python script_test_in.py --experiment imagenetc --resume rn50_daam
python script_test_in.py --experiment imagenetc --resume rn50_moex
python script_test_in.py --experiment imagenetc --resume rvt
