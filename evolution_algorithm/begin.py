# Arman Moussavi


import random
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco_viewer
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
import time
import os.path

warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()





def generate_parent_node_coordinates(z, num_limbs, cube_size):
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


def simulation_settings(timestep, density, viscosity, texture_rgb1, texture_rgb2):

    mujoco = ET.Element('mujoco')
    option = ET.SubElement(mujoco, 'option')
    option.text = f'timestep="{timestep}"'
    option.text = f'density="{density}"'
    option.text = f'viscosity="{viscosity}"'


    asset = ET.SubElement(mujoco, 'asset')
    texture = ET.SubElement(asset, 'texture')
    texture.set('builtin', 'gradient')
    texture.set('height', '256')
    texture.set('rgb1', f'{texture_rgb1}')
    texture.set('rgb2', f'{texture_rgb2}')
    texture.set('type', 'skybox')
    texture.set('width', '256')

    visual = ET.SubElement(mujoco, 'visual')
    headlight = ET.SubElement(visual, 'headlight')
    headlight.set('ambient', '0.5 0.5 0.5')


    return mujoco


def create_mujoco_parent_model(cube_size, node_coordinates, model_name):

    limb_num = range(len(node_coordinates)-1)
    mujoco = simulation_settings(timestep, density, viscosity, texture_rgb1, texture_rgb2)
    worldbody = ET.SubElement(mujoco, 'worldbody')
    light = ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 10", dir="0 0 -1")
    floor = ET.SubElement(worldbody, 'geom', name="floor", type="plane", size="50 50 50", rgba="0.9176 0.8078 0.4156 1.0")
    in_x, in_y, in_z = node_coordinates[0]
    initial_position = f"{in_x} {in_y} {in_z}"
    body = ET.SubElement(worldbody, 'body')
    joint = ET.SubElement(body, 'freejoint', name="body")
    body_shape = ET.SubElement(body, 'geom', name="body", pos=initial_position, type="box", size=f"{cube_size} {cube_size} {cube_size}", rgba="225 0 225 .5", mass="1")

    limb_list = []
    limb_elements = []

    for limb_nums in limb_num:

        if limb_nums + 1 < len(node_coordinates):
            limb_x, limb_y, limb_z = node_coordinates[limb_nums + 1]
            sz_limb_x = (cube_size) if limb_x > cube_size - 0.0001 else (cube_size if limb_x < -cube_size + 0.0001 else cube_size/5)
            sz_limb_y = (cube_size) if limb_y > cube_size - 0.0001 else (cube_size if limb_y < -cube_size + 0.0001 else cube_size/5)
            sz_limb_z = 0.05
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

        limb_elements.append(limb)

    tree = ET.ElementTree(mujoco)
    tree.write(model_name)

    return limb_list, body, mujoco, limb_elements, tree


def write_tree(mujoco,model_name):
    tree = ET.ElementTree(mujoco)
    tree.write(model_name)


def simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer, show_plot):

    model = mj.MjModel.from_xml_path(model_name)
    data = mj.MjData(model)
    actuators = model.nu
    move = np.ones(actuators) * move_factor
    
    data.ctrl[:actuators] = np.abs(move)
    in_x, in_y, in_z = node_coordinates[0]
    initial_body_pos = np.array([in_x, in_y, in_z])
    viewer = None
    trajectory_points = []

    if show_viewer:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    for i in range(steps):
        if i % 20 == 0:
            move *= -1

        current_body_pos = data.qpos[:3]
        trajectory_points.append(current_body_pos.copy())
        data.ctrl[:actuators] = move
        mj.mj_step(model, data)

        # Change camera position and viewing point
        if show_viewer and viewer.is_alive:
            viewer.cam.distance = 40.0  
            viewer.cam.azimuth = 90  


            viewer.cam.elevation = -20  


            viewer.render()


        if show_viewer and viewer.is_alive:
            viewer.render()


    sum_of_absolute_y_displacements = 0

    for i in range(1, len(trajectory_points)):
        _, y_prev, _ = trajectory_points[i-1]
        _, y_curr, _ = trajectory_points[i]
        displacement = abs(y_curr - y_prev)
        sum_of_absolute_y_displacements += displacement

    distance_moved_x = current_body_pos[0]
    distance_weight = 0.9
    deviation_weight = 0.1
    final_body_pos = data.qpos[:3]
    distance_moved_x = final_body_pos[0]
    # deviation_from_straight_path = np.abs(initial_body_pos[1] - final_body_pos[1])

    # Calculate the fitness parameter
    fitness_parameter = distance_weight * distance_moved_x - deviation_weight * sum_of_absolute_y_displacements


    if show_plot:
        if trajectory_points:
            trajectory_points = np.array(trajectory_points)
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], label='Trajectory')
            plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], color='red', label='Final Position', zorder=2)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.savefig('generation_traj.png')
            plt.show()

    if show_viewer:
        viewer.close()

    return fitness_parameter


def test_parents(steps, population_size, max_limbs, z, cube_size, move_factor):
    distances = []
    limb_number = []
    node_coordinates_list =[]

    for iteration in range(population_size):
        random_seed = random.randint(1, 1000)
        random.seed(random_seed)
        num_limbs = random.randint(0, max_limbs)
        node_coordinates = generate_parent_node_coordinates(z, num_limbs, cube_size)
        create_mujoco_parent_model(cube_size, node_coordinates, model_name)
        distance_fitness = simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer=False, show_plot=False)
        distances.append(distance_fitness)
        limb_number.append(num_limbs)
        node_coordinates_list.append(node_coordinates)





    max_distance_index = np.argmax(distances)
    max_distance = distances[max_distance_index]
    num_limbs = limb_number[max_distance_index]
    best_coordinates_list = node_coordinates_list[max_distance_index]

    distance_fitness = max_distance
    node_coordinates = best_coordinates_list



    return distance_fitness, node_coordinates, num_limbs


def select_coordinates_to_mutate(node_coordinates, mutations, cube_size):
    new_coordinates = node_coordinates.copy()  # Make a copy to avoid modifying the original list
    
    selected_coordinates = set()  # Set to keep track of selected coordinates
    
    for _ in range(mutations):
        # Select a coordinate that hasn't been selected before
        available_coordinates = [coord for coord in node_coordinates[1:] if coord not in selected_coordinates]
        if not available_coordinates:
            break  # Break if all coordinates have been selected
        
        selected_coord = random.choice(available_coordinates)
        selected_coordinates.add(selected_coord)  # Add the selected coordinate to the set
        
        selected_x, selected_y, selected_z = selected_coord


        if abs(selected_x) == 1 or abs(selected_y) == 1:
            if abs(selected_x) > abs(selected_y):
                if selected_x > 0:
                    new_point = (selected_x + 3*cube_size, selected_y, selected_z)
                else:
                    new_point = (selected_x - 3*cube_size, selected_y, selected_z)
            else:
                if selected_y < 0:
                    new_point = (selected_x, selected_y - 3*cube_size, selected_z)
                else:
                    new_point = (selected_x, selected_y + 3*cube_size, selected_z)
        else:
            if abs(selected_x) > abs(selected_y):
                if selected_x > 0:
                    new_point = (selected_x + 2*cube_size, selected_y, selected_z)
                else:
                    new_point = (selected_x - 2*cube_size, selected_y, selected_z)
            else:
                if selected_y < 0:
                    new_point = (selected_x, selected_y - 2*cube_size, selected_z)
                else:
                    new_point = (selected_x, selected_y + 2*cube_size, selected_z)

        
        new_coordinates.append(new_point)
    
    return new_coordinates


def test_model(steps, population_size, cube_size, move_factor, node_coordinates):

    distances = []
    node_coordinates_list =[]
    node_coordinates = node_coordinates.copy()

    for iteration in range(population_size):
        mutations = random.randint(0, 10)
        new_coordinates = select_coordinates_to_mutate(node_coordinates, mutations, cube_size)
        create_mujoco_parent_model(cube_size, new_coordinates, model_name)
        distance_fitness = simulate_movement(new_coordinates, steps, move_factor, model_name, show_viewer=False, show_plot=False)
        distances.append(distance_fitness)
        node_coordinates_list.append(new_coordinates)

    max_distance_index = np.argmax(distances)
    max_distance = distances[max_distance_index]
    best_coordinates_list = node_coordinates_list[max_distance_index]

    node_coordinates = best_coordinates_list
    distance_fitness = max_distance




    return distance_fitness, node_coordinates












########################
#######          #######
#######          #######
####### Settings #######
#######          #######
#######          #######
########################
global_seed = 123
random.seed(global_seed) 



# Simulation Settings
########################
steps = 2000
timestep=0.1
density=1000
viscosity=1.002e-3
texture_rgb1='0 1 1'
texture_rgb2='.2 .3 .4'
########################



# Lineage Settings
########################
num_generations = 500
population_size = 100
########################



# Appearance Settings
########################
max_limbs = 10
max_mutations = 10
move_factor = 10
z = 5
cube_size = 1
model_name = 'life.xml'
########################





########################
########################
########################




####################################
##  Run the process of life       ##
####################################

process_of_life = []
life_blueprint = []

# Generation Zero -- Original Parents
distance_fitness, node_coordinates, num_limbs = test_parents(steps, population_size, max_limbs, z, cube_size, move_factor)
process_of_life.append(distance_fitness)
life_blueprint.append(node_coordinates)



# Generations
for generation in range(num_generations):
    distance_fitness, node_coordinates = test_model(steps, population_size, cube_size, move_factor, life_blueprint[-1])
    if distance_fitness > process_of_life[-1]:
        process_of_life.append(distance_fitness)
        life_blueprint.append(node_coordinates)
    else:
        process_of_life.append(process_of_life[-1])
        life_blueprint.append(life_blueprint[-1])





####################################
##     Write files                ##
####################################



end_time = time.time()
elapsed_time = end_time - start_time
timefile = 'execution_time.txt'
if os.path.exists(timefile):
    mode = 'a'
else:
    mode = 'w'

with open(timefile, mode) as f:
    f.write(f'The code took {elapsed_time:.2f} seconds to run.\nGeneration: {num_generations}\nPopulation per generation: {population_size}\nSteps: {steps}\n\n\n')


coordfile = 'blueprints_to_life.txt'
with open(coordfile, 'w') as file:
    for sublist in life_blueprint:
        for item in sublist:
            file.write(f"{item}\n")
        file.write("\n")


    file.write("global_seed {}\n".format(global_seed))
    file.write("steps {}\n".format(steps))
    file.write("timestep {}\n".format(timestep))
    file.write("density {}\n".format(density))
    file.write("viscosity {}\n".format(viscosity))
    file.write("texture_rgb1 {}\n".format(texture_rgb1))
    file.write("texture_rgb2 {}\n".format(texture_rgb2))
    file.write("num_generations {}\n".format(num_generations))
    file.write("population_size {}\n".format(population_size))
    file.write("max_limbs {}\n".format(max_limbs))
    file.write("max_mutations {}\n".format(max_mutations))
    file.write("move_factor {}\n".format(move_factor))
    file.write("z {}\n".format(z))
    file.write("cube_size {}\n".format(cube_size))
    file.write("model_name {}\n".format(model_name))





####################################
##   Plot the process of life     ##
####################################


generations = range(0, num_generations + 1)

plt.figure(figsize=(8, 6))
plt.plot(generations, process_of_life, color='blue', linewidth=2.5, marker='o', markersize=8)
plt.xlabel('Generation', fontsize=14, fontweight='bold')
plt.ylabel('Distance Fitness', fontsize=14, fontweight='bold')
plt.title('The Process of Life', fontsize=16, fontweight='bold')
plt.grid(True) 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('process_of_life_plot.png')
plt.show()





####################################
##     Visualize a generation     ##
####################################

# NOTE: you must close The Process of Life plot to visualize the simulation


generation = -1     # Most fit model


create_mujoco_parent_model(cube_size, life_blueprint[generation], model_name)
distance_fitness = simulate_movement(life_blueprint[generation], steps, move_factor, model_name, show_viewer=True, show_plot=True)



