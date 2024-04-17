num_epochs=600
batch_size=48
num_episodes=50
train_dir=$HOME/train0315
pretrain_ckpt=~/train0312/policy_best.ckpt
ws_path=$(pwd)

# echo "$pretrain_ckpt"
# echo "$train_dir"
# echo $(pwd)

cd $ws_path

python act/train.py --dataset /media/lin/T7/data0314/ --ckpt_dir $train_dir/pretrain --batch_size $batch_size  --num_epochs $num_epochs --num_episodes $num_episodes --pretrain_ckpt $pretrain_ckpt
python act/train.py --dataset /media/lin/T7/data0314/ --ckpt_dir $train_dir/no_pretrain --batch_size $batch_size --num_epochs $num_epochs --num_episodes $num_episodes