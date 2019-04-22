#!/bin/bash
# plan step 5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=5 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=0.5 -end1=0.5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=5 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=1.0 -end1=1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=5 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=0.5 -end1=1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=5 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-0.5 -end1=-0.5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=5 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-1.0 -end1=-1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=5 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-0.5 -end1=-1.0
# plan step 10
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=10 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=0.5 -end1=0.5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=10 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=1.0 -end1=1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=10 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=0.5 -end1=1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=10 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-0.5 -end1=-0.5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=10 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-1.0 -end1=-1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=10 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-0.5 -end1=-1.0
# plan step 15
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=15 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=0.5 -end1=0.5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=15 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=1.0 -end1=1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=15 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=0.5 -end1=1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=15 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-0.5 -end1=-0.5
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=15 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-1.0 -end1=-1.0
python pipeline.py -exp=no_cond22Apr -datasource=point -num_steps=200 -plan_steps=15 -resume_iter=176000 -n_benchmark_exp=10 -cond=False -end1=-0.5 -end1=-1.0

