    
for ((i=0;i<5;i++))
do

    # TwoColors Environment
    
    python test.py --env-name "TwoColors-v1" --algo ppo --clip-param 0.2 --use-gae --log-dir "logs_gridworld/TwoColors/PPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits

    python test.py --env-name "TwoColors-v1" --algo ppo --alpha 0.01 --clip-param 0.2 --use-gae --log-dir "logs_gridworld/TwoColors/SPPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits --augment-type "shifted" --temperature-decay-rate 0.93

    # Diagonal Environment
    
    python test.py --env-name "Diagonal-v1" --algo ppo --clip-param 0.2 --use-gae --log-dir "logs_gridworld/Diagonal/PPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits

    python test.py --env-name "Diagonal-v1" --algo ppo --alpha 0.01 --clip-param 0.2 --use-gae --log-dir "logs_gridworld/Diagonal/SPPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits --augment-type "shifted" --temperature-decay-rate 0.93  

done

# plot the result
python plot.py --log-dir "logs_gridworld" --figure-name "gridworld_1" --multiplots --tiling "symmetric"


