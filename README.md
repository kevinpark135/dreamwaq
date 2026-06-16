This repository builds upon the vanilla "a1" sub-directory of Isaac Lab. The content of this repo should REPLACE "a1" files before it can be used.

CLI commands:

./isaaclab.sh -p ~/research/a1/dreamwaq/train_dreamwaq.py \
  --task Isaac-DreamWaQ-A1-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 1

./isaaclab.sh -p ~/research/a1/dreamwaq/train_baseline.py \
  --task Isaac-DreamWaQ-A1-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 1
