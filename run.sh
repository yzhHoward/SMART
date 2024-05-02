random_port=$(shuf -i 1000-9999 -n 1)
for dataset in 'c12' 'c19' 'mimic_mortality' 'mimic_decompensation' 'mimic_phenotyping' 'mimic_lengthofstay'
do
for seed in 1 42 3407
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 --master_port $random_port \
    main_pretrain.py --dataset $dataset --seed $seed
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 --master_port $random_port \
    main_finetune.py --dataset $dataset --seed $seed
done
done
