
for ((i=0;i<5;i++))
do
    
    python test.py --use-gae --algo ppo --seed $i --log-dir "logs_gridworld/TwoColors/PPO-${i}" --alpha 0 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits --use-clipped-value-loss
    
    python test.py --use-gae --algo ppo --seed $i --log-dir "logs_gridworld/TwoColors/SPPO-${i}" --alpha 3e-7 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits --augment-type 'shifted'
    
    python test.py --use-gae --algo ppo --seed $i --log-dir "logs_gridworld/TwoColors/IPPO-${i}" --alpha 3e-7 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 250000 --use-linear-lr-decay --use-proper-time-limits --augment-type 'invariant'
    
done

