# Code for Model Based Planning with Energy Based Models

Command for training on continual reacher environment:

```
python train.py --exp=reacher_mppi --plan_steps=20 --num_plan_steps=80
--datasource=continual_reacher --noise_sim=80 --mppi --g_coeff=1.0 --a_coeff=0.01  --v_coeff=0.1 --save_interval=300 
--num_env=8 
```

Command for evaluating on continual reacher environment:

```
python train.py --exp=<exp_name> --plan_steps=20 --num_plan_steps=80 --datasource=continual_reacher
--noise_sim=80 --mppi --g_coeff=10.0 --a_coeff=0.01  --v_coeff=0.1 --save_interval=300   --num_env=1 
--resume_iter=<train iteration> --n_benchmark_exp=1 --rl_train=False --train=False
```

Bibtex for paper:

```
@article{Du2019ModelBP,
  title={Model Based Planning with Energy Based Models},
  author={Yilun Du and Toru Lin and Igor Mordatch},
  journal={CORL},
  year={2019}
}
```
