#!/bin/bash

seed=0
epoch=200

python -u ./src/gmixup.py --data_path . --model GIN --dataset PTC_MR \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset PROTEINS \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset IMDB-BINARY \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset NCI1 \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset PTC_MR \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --tail_aug --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset PROTEINS \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --tail_aug --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset IMDB-BINARY \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --tail_aug --epoch=$epoch

python -u ./src/gmixup.py --data_path . --model GIN --dataset NCI1 \
--lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
--aug_ratio 0.15 --aug_num 10  --ge USVT --size_strat --tail_aug
