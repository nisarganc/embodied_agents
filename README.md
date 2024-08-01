# Installation

## Environment 
- source ~/anaconda3/bin/activate
- conda create -n cordinatedeai python=3.9 cmake=3.14.0
- conda activate cordinatedeai

## Habitat Simulator
- conda install habitat-sim withbullet -c conda-forge -c aihabitat

## Dataset and Cordinated task execution package
- git clone --branch stable https://github.com/nisarganc/cordinated_taskexecution.git
- python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path ./downloaded_data
- cd cordinated_taskexecution
- ln -s ~/embodied_agents/downloaded_data data

## Habitat Lab
- cd ..
- git --branch stable clone https://github.com/facebookresearch/habitat-lab.git
- cd habitat-lab
- pip install -e habitat-lab
- pip install typeguard
- pip install jinja2
- pip install -e habitat-baselines
- pip install numpy=1.23.0

# Testing
- habitat-viewer --enable-physics --dataset downloaded_data/replica_cad/replicaCAD.scene_dataset_config.json -- apt_1