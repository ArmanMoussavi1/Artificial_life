# Arman Moussavi


import random
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco_viewer
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



def generate_node_coordinates(z, num_limbs, cube_size):
    coordinates = []

    center_x = center_y = 0
    center_z = z
    coordinates.append((center_x, center_y, center_z))

    for _ in range(num_limbs):
        if random.choice([True, False]):
            random_x = random.choice([cube_size, -cube_size])
            random_y = random.uniform(-cube_size, cube_size)
        else:
            random_x = random.uniform(-cube_size, cube_size)
            random_y = random.choice([cube_size, -cube_size])

        random_z = random.uniform(z - cube_size/2, z + cube_size/2)

        coordinates.append((random_x, random_y, random_z))

    return coordinates



def create_mujoco_model(steps, num_limbs, cube_size, move_factor, node_coordinates, show_viewer=True, show_plot=True):

    limb_num = range(num_limbs)  

    mujoco = ET.Element('mujoco')
    option = ET.SubElement(mujoco, 'option')
    option.text = 'timestep="0.1"'
    option.text = 'density="1000"'
    option.text = 'viscosity="1.002e-3"'

    # Body
    worldbody = ET.SubElement(mujoco, 'worldbody')
    light = ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 10", dir="0 0 1")
    floor = ET.SubElement(worldbody, 'geom', name="floor", type="plane", size="50 50 50", rgba="0 1 0 1")

    in_x, in_y, in_z = node_coordinates[0]

    initial_position = f"{in_x} {in_y} {in_z}"

    body = ET.SubElement(worldbody, 'body')
    joint = ET.SubElement(body, 'freejoint', name="body")
    body_shape = ET.SubElement(body, 'geom', name="body", pos=initial_position, type="box", size=f"{cube_size} {cube_size} {cube_size}", rgba="225 0 225 .5", mass="1")



    limb_list = []

    for limb_nums in limb_num:
        
        if limb_nums + 1 < len(node_coordinates):
            limb_x, limb_y, limb_z = node_coordinates[limb_nums + 1]

            # Adjust size based on conditions
            sz_limb_x = (cube_size) if limb_x > cube_size - 0.0001 else (cube_size if limb_x < -cube_size + 0.0001 else cube_size/5)
            sz_limb_y = (cube_size) if limb_y > cube_size - 0.0001 else (cube_size if limb_y < -cube_size + 0.0001 else cube_size/5)
            sz_limb_z = 0.05

            # Adjust position based on conditions
            limb_x += (cube_size) if limb_x == cube_size else ((-cube_size) if limb_x == -cube_size else 0)
            limb_y += (cube_size) if limb_y == cube_size else ((-cube_size) if limb_y == -cube_size else 0)

            limb_pos = f"{limb_x} {limb_y} {limb_z}"
            limb_size = f"{sz_limb_x} {sz_limb_y} {sz_limb_z}"
            

            limb_list.append({'x': limb_x, 'y': limb_y, 'z': limb_z, 'sz_x': sz_limb_x, 'sz_y': sz_limb_y, 'sz_z': sz_limb_z})
        else:
            None

        limb = ET.SubElement(body, 'body', name=f"leg_{limb_nums}")
        limb_geom = ET.SubElement(limb, 'geom', pos=limb_pos, type="box", size=limb_size, mass="1")

        joint_x = limb_x / 2
        joint_y = limb_y / 2
        joint_z = limb_z - sz_limb_z

        hinge_axis = "0 0 0" 

        if limb_x > cube_size - 0.0001:
            hinge_axis = "0 1 0"
        if limb_y > cube_size - 0.0001:
            hinge_axis = "-1 0 0"
        if limb_x < -cube_size + 0.0001:
            hinge_axis = "0 -1 0"
        elif limb_y < -cube_size + 0.0001:
            hinge_axis = "1 0 0"                        


        limb_joint = ET.SubElement(limb, 'joint', name=f"leg_{limb_nums}_Hinge", pos=f"{joint_x} {joint_y} {joint_z}", axis=hinge_axis, range="-10 10", limited="true")

        actuator = ET.SubElement(mujoco, 'actuator')
        leg1Motor = ET.SubElement(actuator, 'motor', name=f"leg_{limb_nums}_Hinge", gear="50", joint=f"leg_{limb_nums}_Hinge")

    
    name = 'life.xml'
    tree = ET.ElementTree(mujoco)
    tree.write(name)
    move = np.ones(len(limb_list)) * move_factor
    model = mj.MjModel.from_xml_path(name)
    data = mj.MjData(model)
    actuators = model.nu
    data.ctrl[:actuators] = np.abs(move)
    initial_body_pos = np.array([in_x, in_y, in_z])



    viewer = None

    trajectory_points = []


    if show_viewer:
        viewer = mujoco_viewer.MujocoViewer(model, data)



    for i in range(steps):
        if i % 40 == 0:
            move *= -1
        
        current_body_pos = data.qpos[:3]
        trajectory_points.append(current_body_pos.copy())
        
        data.ctrl[:actuators] = move
        mj.mj_step(model, data)

        if show_viewer and viewer.is_alive:
            viewer.render()
            print(f"{i} | {current_body_pos}")

    current_body_pos = data.qpos[:3]
    distance_moved_x = current_body_pos[0]



    if show_plot:
        # Plot the trajectory line
        if trajectory_points:
            trajectory_points = np.array(trajectory_points)
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], label='Trajectory')
            plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], color='red', label='Final Position', zorder=2)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.show()


    if show_viewer:
        viewer.close()

    return distance_moved_x




def test_bodies(steps, num_iterations, max_limbs, z, cube_size, move_factor):
    distances = []
    random_seeds = []
    limb_number = []


    for iteration in range(num_iterations):
        random_seed = random.randint(1, 1000)
        random.seed(random_seed)

        num_limbs = random.randint(0, max_limbs)

        # Generate and return the coordinates
        node_coordinates = generate_node_coordinates(z, num_limbs, cube_size)

        # Create and run the mujoco model, get the distance moved
        distance_fitness = create_mujoco_model(steps, num_limbs, cube_size, move_factor, node_coordinates, show_viewer=test_viewer, show_plot=test_plot)

        # Store the distance in the list
        distances.append(distance_fitness)
        random_seeds.append(random_seed)
        limb_number.append(num_limbs)




    max_distance_index = np.argmax(distances)
    max_distance = distances[max_distance_index]
    max_distance_random_seed = random_seeds[max_distance_index]
    num_limbs = limb_number[max_distance_index]


    random.seed()
    random.seed(max_distance_random_seed)

    node_coordinates = generate_node_coordinates(z, num_limbs, cube_size)
    distance = create_mujoco_model(steps, num_limbs, cube_size, move_factor, node_coordinates, show_viewer=best_viewer, show_plot=best_plot)

    print(f"Distance Traveled in the X-Direction: {distance}, Corresponding Random Seed: {max_distance_random_seed}")

    return max_distance_random_seed




# Simulation inputs

random.seed(1) 

num_iterations = 100
steps = 1000
max_limbs = 10

z = 3
cube_size = 1
move_factor = 10


# Output options
test_viewer=False
test_plot=False
best_viewer=True
best_plot=True



max_distance_random_seed = test_bodies(steps, num_iterations, max_limbs, z, cube_size, move_factor)


