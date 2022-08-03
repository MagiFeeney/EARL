env = ("HalfCheetah", "Hopper", "Walker2d", "Swimmer", "Ant", "Humanoid")
algos = ("sac", "ppo", "sppo", "ippo", "trpo", "strpo", "itrpo", "td3")

for ((i=0;i<3;i+=1))
do
    for index in "${!envs[@]}"
    do

	python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --use-gae --log-dir "logs/ppo/${env[$index]}-$i/" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --use-clipped-value-loss

	python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --alpha 3e-7 --use-gae --log-dir "logs/sppo/${env[$index]}-$i/" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --augmented-type "shifted"

	python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --alpha 3e-7 --use-gae --log-dir "logs/ippo/${env[$index]}-$i/" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --augmented-type "invariant"
	
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --log-dir "logs/trpo/${env[$index]}-$i/" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.995 --gae-lambda 0.97 --num-env-steps 1000000 --max-kl 0.01 --damping 1e-1 --l2-reg 1e-3
	
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --alpha 3e-7 --log-dir "logs/strpo/${env[$index]}-$i/" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.995 --gae-lambda 0.97 --num-env-steps 1000000 --max-kl 0.01 --damping 1e-1 --l2-reg 1e-3 --augmented-type "shifted"

	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --alpha 3e-7 --log-dir "logs/itrpo/${env[$index]}-$i/" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.995 --gae-lambda 0.97 --num-env-steps 1000000 --max-kl 0.01 --damping 1e-1 --l2-reg 1e-3 --augmented-type "invariant"	     	
	
	python baselines.py --env-name "${env[$index]}-v3" --algo sac --seed $i --num-env-steps 1000000 --log-dir "logs/sac/${env[$index]}-$i/"
	
	python baselines.py --env-name "${env[$index]}-v3" --algo td3 --seed $i --num-env-steps 1000000 --log-dir "logs/td3/${env[$index]}-$i/"
	
    done
done
