export PYTHONPATH=$PYTHONPATH:$(pwd)
python script_test_c10.py --experiment cifar10  --resume rn26_gn
python script_test_c10.py --experiment cifar101 --resume rn26_gn
# the following command takes much longer to run
python script_test_c10.py --experiment cifar10c --resume rn26_gn
