export PYTHONPATH=$PYTHONPATH:../../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../../RL-Robot-Env/
export PYTHONPATH=$PYTHONPATH:../../../HIL-RL
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889



task_name=close_trashbin_franka_1028
mkdir -p experiments/${task_name}
cd experiments/${task_name}


python3 ../../collect_data.py robot_type@_global_=franka task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false
