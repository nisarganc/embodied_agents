import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
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

def init_sim(data_path, agent_dict):

    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    # Set up an example scene
    sim_cfg.scene = os.path.join(data_path, "replica_cad/configs/scenes/apt_5.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "replica_cad/replicaCAD.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]

    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    
    # Create the scene
    sim = RearrangeSim(cfg)

    # This is needed to initialize the agents
    sim.agents_mgr.on_new_scene()

    # For this tutorial, we will also add an extra camera that will be used for third person recording.
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"

    # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
    sim.add_sensor(camera_sensor_spec, 0)

    return sim


################################################
if __name__ == "__main__":

    # dataset path
    import git, os
    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    data_path = os.path.join(dir_path, "downloaded_data")
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

    init_pos = mn.Vector3(2,0,5)
    art_agent = sim.articulated_agent
    # We will see later about this
    art_agent.sim_obj.motion_type = MotionType.KINEMATIC
    print("Current agent position:", art_agent.base_pos)
    art_agent.base_pos = init_pos 
    print("New agent position:", art_agent.base_pos)
    # We take a step to update agent position
    _ = sim.step({})

    observations = sim.get_sensor_observations()
    print(observations.keys())

    observations = []
    num_iter = 100
    pos_delta = mn.Vector3(0.02,0,0)
    rot_delta = np.pi / (8 * num_iter)
    art_agent.base_pos = init_pos

    sim.reset()
    # set_fixed_camera(sim)
    for _ in range(num_iter):
        # TODO: this actually seems to give issues...
        art_agent.base_pos = art_agent.base_pos + pos_delta
        art_agent.base_rot = art_agent.base_rot + rot_delta
        sim.step({})
        observations.append(sim.get_sensor_observations())

    vut.make_video(
        observations,
        "head_rgb",
        "color",
        "video_1",
        open_vid=False,
    )
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "video_2",
        open_vid=False,
    )    