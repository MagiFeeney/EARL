declare -a env=("HalfCheetah" "Hopper" "Walker2d" "Swimmer" "Ant" "Humanoid")

for index in "${!env[@]}"
do
    for ((i=0;i<5;i+=1))
    do
	
	# PPO: Baseline
        python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --use-gae --log-dir "logs/${env[$index]}/PPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits	

	# PPO: Reward shaping with entropy
        python main.py --env-name "${env[$index]}-v3" --algo ppo --alpha 0.01 --clip-param 0.2 --use-gae --log-dir "logs/${env[$index]}/SPPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --augment-type "shifted" --temperature-decay-rate 0.93 --epochs-drop 10

	# PPO: Bootstrapping with state value function
        python main.py --env-name "${env[$index]}-v3" --algo ppo --alpha 0.01 --clip-param 0.2 --use-gae --log-dir "logs/${env[$index]}/BPPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --augment-type "bootstrap" --temperature-decay-rate 0.93 --epochs-drop 10

	# TRPO: Baseline
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --log-dir "logs/${env[$index]}/TRPO-${i}" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --max-kl 0.1 --damping 1e-1 --l2-reg 1e-3 

	# TRPO: Reward shaping with entropy
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --alpha 0.01 --log-dir "logs/${env[$index]}/STRPO-${i}" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --max-kl 0.1 --damping 1e-1 --l2-reg 1e-3 --augment-type "shifted" --temperature-decay-rate 0.93 --epochs-drop 10
	
	# TRPO: Bootstrapping with state value function
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --alpha 0.01 --log-dir "logs/${env[$index]}/BTRPO-${i}" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --max-kl 0.1 --damping 1e-1 --l2-reg 1e-3 --augment-type "bootstrap" --temperature-decay-rate 0.93 --epochs-drop 10

    done
done

# Plot the result
python plot.py --log-dir "logs" --figure-name "mujoco_1" --multiplots --legend-last --tiling "symmetric"
