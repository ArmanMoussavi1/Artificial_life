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


def create_mujoco_parent_model(num_limbs, cube_size, node_coordinates, model_name):
    limb_num = range(num_limbs)

    mujoco = ET.Element('mujoco')
    option = ET.SubElement(mujoco, 'option')
    option.text = 'timestep="0.1"'
    option.text = 'density="1000"'
    option.text = 'viscosity="1.002e-3"'
    worldbody = ET.SubElement(mujoco, 'worldbody')
    light = ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 10", dir="0 0 1")
    floor = ET.SubElement(worldbody, 'geom', name="floor", type="plane", size="50 50 50", rgba="0 1 0 1")
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



def build_parent(num_limbs, cube_size, node_coordinates, model_name):


    global limb_num, mujoco, option, worldbody, light, floor, body, joint, body_shape, limb_list, limb_elements





    limb_num = range(num_limbs)
    
    mujoco = ET.Element('mujoco')
    option = ET.SubElement(mujoco, 'option')
    option.text = 'timestep="0.1"'
    option.text = 'density="1000"'
    option.text = 'viscosity="1.002e-3"'
    worldbody = ET.SubElement(mujoco, 'worldbody')
    light = ET.SubElement(worldbody, 'light', diffuse=".5 .5 .5", pos="0 0 10", dir="0 0 1")
    floor = ET.SubElement(worldbody, 'geom', name="floor", type="plane", size="50 50 50", rgba="0 1 0 1")
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


    return limb_elements, limb_list, mujoco




def mutate_parent(num_limbs, cube_size, node_coordinates, model_name, mutations, limb_elements, limb_list, mujoco, generation):


    global mutated_limbs, mutation_num, selected_limb, mut_limb, mut_limb_joint, mut_actuator, mut_leg_Motor


    mutated_limbs = set() 

    for mutation_num in range(mutations):
        selected_limb = random.choice([limb for limb in limb_elements if limb not in mutated_limbs])
        mutated_limbs.add(selected_limb)
        selected_limb_size = selected_limb.find('geom').attrib['size']
        selected_limb_size = [float(coord) for coord in selected_limb_size.split()]
        selected_limb_pos = selected_limb.find('geom').attrib['pos']
        selected_limb_pos = [float(coord) for coord in selected_limb_pos.split()]
        # print(selected_limb_pos)
        mut_limb_sz_x, mut_limb_sz_y, mut_limb_sz_z = selected_limb_size
        new_limb_x, new_limb_y, new_limb_z = selected_limb_pos
        if abs(new_limb_x) > abs(new_limb_y):
            if new_limb_x > 0:
                new_point = (new_limb_x + 2*mut_limb_sz_x, new_limb_y, new_limb_z)
            else:
                new_point = (new_limb_x - 2*mut_limb_sz_x, new_limb_y, new_limb_z)
        else:
            if new_limb_y < 0:
                new_point = (new_limb_x, new_limb_y - 2*mut_limb_sz_y, new_limb_z)
            else:
                new_point = (new_limb_x, new_limb_y + 2*mut_limb_sz_y, new_limb_z)


        mut_limb_x, mut_limb_y, mut_limb_z = new_point


        limb_pos = f"{mut_limb_x} {mut_limb_y} {mut_limb_z}"
        limb_size = f"{mut_limb_sz_x} {mut_limb_sz_y} {mut_limb_sz_z}"




        mut_limb = ET.SubElement(selected_limb, 'body', name=f"gen_{generation}_mut_leg_{len(limb_list) + mutation_num}")


        limb_elements.append(mut_limb)


        limb_geom = ET.SubElement(mut_limb, 'geom', pos=limb_pos, type="box", size=limb_size, mass="1")


        selected_limb_hinge_axis = selected_limb.find('joint').attrib['axis']




        if abs(new_limb_x) > abs(new_limb_y):
            if new_limb_x > 0:
                new_point = (new_limb_x + mut_limb_sz_x, new_limb_y, new_limb_z)
            else:
                new_point = (new_limb_x - mut_limb_sz_x, new_limb_y, new_limb_z)

        else:
            if new_limb_y < 0:
                new_point = (new_limb_x, new_limb_y - mut_limb_sz_y, new_limb_z)
            else:
                new_point = (new_limb_x, new_limb_y + mut_limb_sz_y, new_limb_z)

        new_limb_x, new_limb_y, new_limb_z = new_point



        mut_limb_joint = ET.SubElement(mut_limb, 'joint', name=f"gen_{generation}_mut_leg_{len(limb_list) + mutation_num}_Hinge", pos=f"{new_limb_x} {new_limb_y} {new_limb_z}", axis=selected_limb_hinge_axis, range="-10 10", limited="true")

        mut_actuator = ET.SubElement(mujoco, 'actuator')

        mut_leg_Motor = ET.SubElement(mut_actuator, 'motor', name=f"gen_{generation}_mut_leg_{len(limb_list) + mutation_num}_Hinge", gear="50", joint=f"gen_{generation}_mut_leg_{len(limb_list) + mutation_num}_Hinge")







    return limb_list, mujoco, limb_elements





def write_tree(mujoco,model_name):
    tree = ET.ElementTree(mujoco)
    tree.write(model_name)




def simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer, show_plot):

    model = mj.MjModel.from_xml_path(model_name)
    data = mj.MjData(model)
    actuators = model.nu


    # Ensure move has the same length as the number of actuators
    move = np.ones(actuators) * move_factor
    
    data.ctrl[:actuators] = np.abs(move)
    in_x, in_y, in_z = node_coordinates[0]
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
            # print(f"{i} | {current_body_pos}")

    current_body_pos = data.qpos[:3]
    distance_moved_x = current_body_pos[0]

    # print(distance_moved_x)

    if show_plot:
        if trajectory_points:
            trajectory_points = np.array(trajectory_points)
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], label='Trajectory')
            plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], color='red', label='Final Position', zorder=2)
            # print(trajectory_points[-1, 0], trajectory_points[-1, 1])
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.show()

    if show_viewer:
        viewer.close()

    # print(distance_moved_x)

    return distance_moved_x



def test_parents(steps, num_iterations, max_limbs, z, cube_size, move_factor):
    distances = []
    random_seeds = []
    limb_number = []


    for iteration in range(num_iterations):
        random_seed = random.randint(1, 1000)
        random.seed(random_seed)

        num_limbs = random.randint(0, max_limbs)

        # Generate and return the coordinates
        node_coordinates = generate_parent_node_coordinates(z, num_limbs, cube_size)

        create_mujoco_parent_model(num_limbs, cube_size, node_coordinates, model_name)

        # print("test")
        # Create and run the mujoco model, get the distance moved
        distance_fitness = simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer=test_viewer, show_plot=test_plot)
        

        # Store the distance in the list
        distances.append(distance_fitness)
        random_seeds.append(random_seed)
        limb_number.append(num_limbs)




    max_distance_index = np.argmax(distances)
    max_distance = distances[max_distance_index]
    max_distance_random_seed = random_seeds[max_distance_index]
    num_limbs = limb_number[max_distance_index]


    # print(distances)



    random.seed()
    random.seed(max_distance_random_seed)

    num_limbs = random.randint(0, max_limbs)

    # Generate and return the coordinates
    node_coordinates = generate_parent_node_coordinates(z, num_limbs, cube_size)

    create_mujoco_parent_model(num_limbs, cube_size, node_coordinates, model_name)

    # print("test")
    # Create and run the mujoco model, get the distance moved
    distance_fitness = simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer=best_viewer, show_plot=best_plot)



    print(f"Distance Parent Traveled in the X-Direction: {distance_fitness}, Corresponding Random Seed: {max_distance_random_seed}")

    return max_distance_random_seed, num_limbs




def test_generations(max_mutations, steps, num_iterations, max_limbs, z, cube_size, move_factor, max_distance_random_seed_parent, parent_node_coordinates):
    gen_distances = []
    gen_random_seeds = []
    gen_limb_number = []
    gen_mutation_list = []
    gen_generation_list =[]





    random.seed(max_distance_random_seed_parent)
    num_limbs = random.randint(0, max_limbs)

    # print(max_distance_random_seed_parent)

    node_coordinates = generate_parent_node_coordinates(z, num_limbs, cube_size)
    limb_elements, limb_list, mujoco = build_parent(num_limbs, cube_size, node_coordinates, model_name)

    # print(max_distance_random_seed_parent)

    random.seed(global_seed)


    for iteration in range(num_iterations):

        random_seed = random.randint(1, 1000)

        random.seed(random_seed)
        mutations = random.randint(0, max_mutations)
        # print(random_seed)

        generation = iteration

        mutate_parent(num_limbs, cube_size, node_coordinates, model_name, mutations, limb_elements, limb_list, mujoco, generation)

        # print(f"\n\n{limb_elements}")
        write_tree(mujoco,model_name)
        # print("test")

        distance_fitness = simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer=test_viewer, show_plot=test_plot)

        # Store the distance in the list
        gen_distances.append(distance_fitness)
        gen_random_seeds.append(random_seed)
        gen_limb_number.append(num_limbs)
        gen_mutation_list.append(mutations)
        gen_generation_list.append(generation)


    # print(gen_distances)
    # print(gen_random_seeds)
    # print(gen_limb_number)
    # print(gen_mutation_list)
    # print(gen_generation_list)
        


    gen_max_distance_index = np.argmax(gen_distances)
    gen_max_distance = gen_distances[gen_max_distance_index]
    gen_max_distance_random_seed = gen_random_seeds[gen_max_distance_index]
    gen_num_limbs = gen_limb_number[gen_max_distance_index]
    mutations = gen_mutation_list[gen_max_distance_index]
    generation = gen_generation_list[gen_max_distance_index]

    # print(gen_max_distance_index)
    # print(gen_max_distance)
    # print(gen_max_distance_random_seed)
    # print(gen_num_limbs)
    # print(mutations)
    # print(generation)
    # print(gen_random_seeds)


    random.seed()
    random.seed(max_distance_random_seed_parent)

    # print(max_distance_random_seed_parent)
    num_limbs = random.randint(0, max_limbs)


    node_coordinates = generate_parent_node_coordinates(z, num_limbs, cube_size)
    limb_elements, limb_list, mujoco = build_parent(num_limbs, cube_size, node_coordinates, model_name)

    random.seed()
    random.seed(gen_max_distance_random_seed)


    

    # print(num_limbs, mutations, generation)


    
    # mutate_parent(num_limbs, cube_size, node_coordinates, model_name, mutations, limb_elements, limb_list, mujoco, generation)
    # write_tree(mujoco,model_name)

    # distance = simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer=best_viewer, show_plot=best_plot)




    mutations = random.randint(0, max_mutations)
    # print(random_seed)

    generation = iteration


    # print("%%%%%%%%%%%%%%%")
    # print( limb_elements)

    mutate_parent(num_limbs, cube_size, node_coordinates, model_name, mutations, limb_elements, limb_list, mujoco, generation)
    write_tree(mujoco,model_name)
    # print("test")

    distance = simulate_movement(node_coordinates, steps, move_factor, model_name, show_viewer=best_viewer, show_plot=best_plot)







    # print("real")


    # print(distance)


    # print(f"Distance Traveled in the X-Direction: {distance}, Corresponding Random Seed: {max_distance_random_seed}")

    return gen_max_distance_random_seed, mutations, distance













# Simulation inputs


global_seed = 1
random.seed(global_seed) 





num_iterations = 10
steps = 500

max_limbs = 20
max_mutations = 10




z = 3
cube_size = 1
move_factor = 10
model_name = 'life.xml'


# Output options
test_viewer=False
test_plot=False
best_viewer=True
best_plot=True





print("\n*************************")
print("please wait: parent competition")
print("*************************")

max_distance_random_seed_parent, num_limbs = test_parents(steps, num_iterations, max_limbs, z, cube_size, move_factor)
random.seed(max_distance_random_seed_parent)
num_limbs = random.randint(0, max_limbs)
parent_node_coordinates = generate_parent_node_coordinates(z, num_limbs, cube_size)




print("\n*************************")
print("please wait: first generation competition")
print("*************************")

max_distance_random_seed_generation, mutation_nums, distance = test_generations(max_mutations, steps, num_iterations, max_limbs, z, cube_size, move_factor, max_distance_random_seed_parent, parent_node_coordinates)

print("\n\n\n\n\n\n")
print("Parent seed:", max_distance_random_seed_parent)
print("Child seed:", max_distance_random_seed_generation)
print("Number of limbs:", num_limbs)
print("Number of Mutations:", mutation_nums)
print("Child Distance Traveled in the X-Direction::", distance)


