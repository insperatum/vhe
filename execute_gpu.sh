#!/bin/sh

#SBATCH --qos=tenenbaum
#SBATCH --time=1000
#SBATCH --mem=20G
#SBATCH --job-name=vhe
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:titan-x:1


#export PATH=/om/user/mnye/miniconda3/bin/:$PATH
source activate /om/user/mnye/vhe/envs/default/
#cd /om/user/mnye/vhe
#anaconda-project run which python
$@
