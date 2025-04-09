# CA-ICRL
This is the code for ICML paper: "Confidence Aware Inverse Constrained Reinforcement Learning". The full paper is on [arXiv](https://arxiv.org/pdf/2406.16782). 






## Setup Python Virtual Environment
1. Make sure you have [downloaded & installed (mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) before proceeding.
2. Create conda environment and install the packages:
```
mkdir ./save_model
mkdir ./evaluate_model
conda env create -n cn-py37 python=3.7 -f python_environment.yml
pip install --upgrade setuptools==66
pip install --upgrade gym==0.21
conda activate cn-py37
```
3. Install [Pytorch (version==1.21.1)](https://pytorch.org/) in the conda env.


Now just run the appropriate files to run CA-ICRL and the other baselines. 

To run the virtual environment, you need to set up MuJoCo.
1. Download the MuJoCo version 2.1 binaries for Linux or OSX.
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Install and use [mujoco-py](https://github.com/openai/mujoco-py).
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment

# (optional) if mujoco dir need to changed
export MUJOCO_PY_MUJOCO_PATH=YOUR_MUJOCO_DIR/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_MUJOCO_DIR/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

To run the realistic environments you need to setup commonroad. 


```
sudo apt-get update
sudo apt-get install build-essential make cmake

# option 1: Install with sudo rights (cn-py37 is the name of conda environment).
cd ./commonroad_environment
bash ./scripts/install.sh -e cn-py37

# Option 2: Install without sudo rights
bash ./commonroad_environment/scripts/install.sh -e cn-py37 --no-root
```



### Important Notice
Throughout this section, we will use the ```Blocked Half-cheetah``` environment as an example,
for using other environments (including ```Blocked Ant```, ```Biased Pendulumn```, ```Blocked Walker```, ```Blocked Swimmer```, ```HighD Velocity```, ```HighD Distance```, please refer to their configs in this [dir](./config/))

###  Step 2: Train expert agents.
Note that the expert agent is to generate demonstration data (see the step 3 below).
```
# step in the dir containing the "main" files.
cd ./interface/

# run PPO without knowing the constraint
python train_policy.py ../config/mujuco_BlockedHalfCheetah/train_ppo_HCWithPos-v0.yaml -n 5 -s 1

# run PPO-Lag knowing the ground-truth
python train_policy.py ../config/mujuco_BlockedHalfCheetah/train_ppo_lag_HCWithPos-v0.yaml -n 5 -s 1
```

###  Step 3: Generate the expert demonstration.

```
# step in the dir containing the "main" files.
cd ./interface/

# run data generation
python generate_data_for_constraint_inference.py -n 5 -mn train_ppo_lag_HCWithPos-v0-multi_env-Apr-21-2022-04:49-seed_1 -tn PPO-Lag-HC -ct no-constraint -rn 1
```

### Step 4: Run the ICLR algorithms
We use the ```Blocked Half-Cheetah``` environment as an example (also see the notice above).
```
# step in the dir containing the "main" files.
cd ./interface/

# run GACL
python train_gail.py ../config/mujuco_BlockedHalfCheetah/train_GAIL_HCWithPos-v0.yaml -n 5 -s 1

# run BC2L
python train_icrl.py ../config/mujuco_BlockedHalfCheetah/train_Binary_HCWithPos-v0.yaml -n 5 -s 1

# run ICRL
python train_icrl.py ../config/mujuco_BlockedHalfCheetah/train_ICRL_HCWithPos-v0.yaml -n 5 -s 1

# run VICRL
python train_icrl.py ../config/mujuco_BlockedHalfCheetah/train_VICRL_HCWithPos-v0.yaml -n 5 -s 1

# run CAICRL
python train_icrl.py ../config/mujuco_BlockedHalfCheetah/train_CAICRL_HCWithPos-v0.yaml -n 5 -s 1
```


## Note

This is research code and will not be actively maintained. Please send an email to ***sriram.subramanian@vectorinstitute.ai*** for questions or comments.



## Paper citation

If you found this helpful, please cite the following paper:

<pre>



@InProceedings{SriramCAICRL2024,
  title = 	 {Confidence Aware Inverse Constrained Reinforcement Learning},
  author = 	 {Subramanian, Sriram Ganapathi and Liu, Guiliang and Elmahgiubi, Mohammed and Rezaee, Kasra and Poupart, Pascal} 
  booktitle = 	 {Proceedings of the International Conference on Machine Learning (ICML 2024)},
  year = 	 {2024},
  address = 	 {Vienna, Austria},
  month = 	 {21 Jul -- 27 Jul},
  publisher = 	 {PMLR}
}
</pre>

