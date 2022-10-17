# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```
## Reproduce
### intent
```shell
python3 train_intent.py --data_dir ${data_dir} --cache_dir ${cache_dir} --ckpt_dir ${ckpt_dir} --num_workers ${num CPU cores} --device ${cude device}
```

### slot
```shell
python3 train_slot.py --data_dir ${data_dir} --cache_dir ${cache_dir} --ckpt_dir ${ckpt_dir} --num_workers ${num CPU cores} --device ${cude device}
```

### seqeval
```shell
python3 eval_slot.py --test_file ${test file} --data_dir ${data_dir} --cache_dir ${cache_dir} --ckpt_dir ${model.pt} --device ${device}
```

## Inference
First download the models
```shell
bash ./download.sh
```
### intent
```shell
bash ./intent_cls.sh ${path_to_test.json} ${path_to_pred.csc}
```

### slot
```shell
bash ./slot_tag.sh ${path_to_test.json} ${path_to_pred.csc}
```