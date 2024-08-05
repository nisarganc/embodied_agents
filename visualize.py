import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf

from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
import gzip
import json

def init_sim(data_path, agent_dict):

    # Scene Configs
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True
    sim_cfg.scene = os.path.join(data_path, "replica_cad/configs/scenes/apt_4.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "replica_cad/replicaCAD.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]
    
    # Agent configs
    cfg = OmegaConf.create(sim_cfg)
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    
    # Create the Rearrange Simulator 
    sim = RearrangeSim(cfg)

    # 3rd person camera sensor
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"
    sim.add_sensor(camera_sensor_spec, 0)

    #initialize the agents
    sim.agents_mgr.on_new_scene()
    return sim


################################################
if __name__ == "__main__":

    # dataset path
    import git, os
    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    data_path = os.path.join(dir_path, "data")
    os.chdir(dir_path)

    # Define the agent configuration (URDF)
    main_agent_config = AgentConfig()
    urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
    main_agent_config.articulated_agent_urdf = urdf_path
    main_agent_config.articulated_agent_type = "FetchRobot"

    # Dictionary with names of agents and their corresponding agent configuration
    agent_dict = {"main_agent": main_agent_config}
    # Add sensors to the agent urdf
    main_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
    }
    sim = init_sim(data_path, agent_dict)

    ############ MOVE ACTION ############
    
    art_agent = sim.articulated_agent
    # print(type(art_agent))

    art_agent.sim_obj.motion_type = MotionType.DYNAMIC
    final_pos = mn.Vector3(2,0,5)

    print("Current agent position:", art_agent.base_pos)
    print("New agent position:", final_pos)

    observations = []
    num_iter = 100
    pos_delta = mn.Vector3(0.02,0,0)
    print("Delta:", pos_delta)
    rot_delta = np.pi / (1 * num_iter)
    print("Delta:", rot_delta)

    for _ in range(num_iter):
        art_agent.base_pos = art_agent.base_pos + pos_delta
        art_agent.base_rot = art_agent.base_rot + rot_delta
        sim.step({})
        sim.step_physics(1./60)
        observations.append(sim.get_sensor_observations())

    # vut.make_video(
    #     observations,
    #     "head_rgb",
    #     "color",
    #     "results/head_rgb_moveagent_video",
    #     open_vid=False,
    # )
    # vut.make_video(
    #     observations,
    #     "third_rgb",
    #     "color",
    #     "results/third_rgb_moveagent_video",
    #     open_vid=False,
    # ) 

    ############ ARM JOINT ACTIONs ############  

    sim.reset()
    observations = []
    print("Arm joint limits:", art_agent.arm_joint_limits)

    lower_limit = art_agent.arm_joint_limits[0].copy()
    lower_limit[lower_limit == -np.inf] = 0

    upper_limit = art_agent.arm_joint_limits[1].copy()
    upper_limit[upper_limit == np.inf] = 0

    for i in range(num_iter):
        alpha = i/num_iter
        current_joints = upper_limit * alpha + lower_limit * (1 - alpha)
        art_agent.arm_joint_pos = current_joints
        sim.step({})
        observations.append(sim.get_sensor_observations())
        if i in [0, num_iter-1]:
            print(f"Step {i}:")
            print("Arm joint positions:", art_agent.arm_joint_pos)
            print("Arm end effector translation:", art_agent.ee_transform().translation)
            print(art_agent.sim_obj.joint_positions)

    # vut.make_video(
    #     observations,
    #     "third_rgb",
    #     "color",
    #     "results/third_rgb_jointaction_video",
    #     open_vid=False,
    # )

    ####### DYNAMIC UPDATION ########

    # We will initialize the agent 0.3 meters away from the floor and let it fall
    sim.reset()
    art_agent._fixed_base = False
    sim.agents_mgr.on_new_scene()

    # The base is not fixed anymore
    art_agent.sim_obj.motion_type = MotionType.DYNAMIC


    art_agent.base_pos = art_agent.base_pos + mn.Vector3(0,1.5,0)

    _ = sim.step({})
    observations = []
    fps = 60 # Default value for make video
    dt = 1./fps
    for _ in range(120):    
        sim.step_physics(dt)
        observations.append(sim.get_sensor_observations())
        
    # vut.make_video(
    #     observations,
    #     "third_rgb",
    #     "color",
    #     "results/Dynamic_falling_video",
    #     open_vid=False,
    # )

    ######################## OBJECTS IN SCENE ####################

    # Load the dataset
    episode_file = os.path.join(data_path, "datasets/replica_cad/rearrange/v1/minival/rearrange_easy.json.gz")
    with gzip.open(episode_file, "rt") as f: 
        episode_files = json.loads(f.read())

    # Get the first episode
    episode = episode_files["episodes"][0]
    rearrange_episode = RearrangeEpisode(**episode)

    art_agent = sim.articulated_agent
    art_agent._fixed_base = True
    sim.agents_mgr.on_new_scene()

    sim.reconfigure(sim.habitat_config, ep_info=rearrange_episode)
    sim.reset()    

    aom = sim.get_articulated_object_manager()
    rom = sim.get_rigid_object_manager()

    # We can query the articulated and rigid objects
    print("List of articulated objects:")
    for handle, ao in aom.get_objects_by_handle_substring().items():
        print(handle, ", id:", aom.get_object_id_by_handle(handle))

    print("\nList of rigid objects:")
    obj_ids = []
    for handle, ro in rom.get_objects_by_handle_substring().items():
        if ro.awake:
            print(handle, ",id:", ro.object_id)
            obj_ids.append(ro.object_id)                

    ########################### TELEPORTATION AND GRASPING ######################

    sim.reset()
    art_agent._fixed_base = True
    art_agent.sim_obj.motion_type = MotionType.KINEMATIC

    # 002_master_chef_can_:0000 ,id: 82
    obj_id = int(82)
    first_object = rom.get_object_by_id(obj_id)
    sim.agents_mgr.on_new_scene()
    print("Agent's position", sim.articulated_agent.base_pos)


    object_trans = first_object.translation
    print("Object's position", first_object.handle, "is in", object_trans)

    sample = sim.pathfinder.get_random_navigable_point_near(
        circle_center=object_trans, radius=1.0, island_index=-1
    )
    print("Sampled position:", sample)

    vec_sample_obj = object_trans - sample
    angle_sample_obj = np.arctan2(-vec_sample_obj[2], vec_sample_obj[0])
    print("Angle between object and sample:", angle_sample_obj)

    sim.articulated_agent.base_pos = sample
    sim.articulated_agent.base_rot = angle_sample_obj
    obs = sim.step({})

    plt.imsave('results/output.png', obs['third_rgb'])

    # grasp manager to interact with the object:
    agent_id = 0
    grasp_manager = sim.agents_mgr[agent_id].grasp_mgrs[0]
    grasp_manager.snap_to_obj(obj_id)
    obs = sim.step({})
    plt.imsave('results/output2.png', obs['third_rgb'])


    ########################### DROP OBJECT ######################
    num_iter = 10
    observations = []
    #sim.articulated_agent.base_pos = sample
    for _ in range(num_iter):    
        forward_vec = art_agent.base_transformation.transform_vector(mn.Vector3(1,0,0))
        art_agent.base_pos = art_agent.base_pos + forward_vec * 0.02
        art_agent.base_rot = np.pi / (1 * num_iter)
        observations.append(sim.step({}))
        
    # Remove the object
    grasp_manager.desnap()
    for _ in range(20):
        observations.append(sim.step({}))
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "results/drop_video",
        open_vid=False,
    )