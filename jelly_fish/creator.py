import mujoco
import mujoco_viewer
import numpy as np
import random





####################################
### Inputs ###

gravity = "-1"
central_body = "box"
body_options = "capsule, ellipsoid, cylinder"
central_bod_color = "225 0 225 1"
limb_color = "0 0 1 1"


limb_nums = random.randint(1, 10)
limb_adds = random.randint(1, 20)
angle = random.randint(0, 120)
####################################







####################################
with open(f"beginning_of_life.xml", 'w') as file:
    file.write("")
####################################



create_model = f"""
<mujoco model="the_beginning">
<visual>
<headlight ambient="0.5 0.5 0.5"/>
</visual>
<option timestep="0.01" density="1000" viscosity="1.002e-3"/>


"""

with open(f"beginning_of_life.xml", 'a') as file:
    file.write(create_model)





body_type = random.choice(body_options.split(', '))

start_body = f"""
<worldbody>
<light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
<geom type="plane" size="20 20 0.1" rgba="0 .8 .2 1"/>
"""

with open(f"beginning_of_life.xml", 'a') as file:
    file.write(start_body)







central_body_size = random.randint(100,200)*0.01

central_body = f"""
<body pos="0 0 30"  euler="0 0 0">
<joint type="free" axis="-1 0 0" pos="0 0 -0.5"/>
<geom type="{central_body}" size="{central_body_size}" fromto="0 0 0 0 0 1" rgba="{central_bod_color}" mass="1"/>
"""

with open(f"beginning_of_life.xml", 'a') as file:
    file.write(central_body)





for limb_num in range(1, limb_nums + 1):

    pos_z = 0
    pos_x = random.uniform(-central_body_size/2, central_body_size/2)  # Random x-position between -central_body_size and central_body_size
    pos_y = random.uniform(-central_body_size/2, central_body_size/2)
    limb_size = 0.1 #random.randint(50,100)*0.01
    bod_limb = f"""

    <body pos="{pos_x} {pos_y} {pos_z}" euler="180 0 0">
        <joint name="{limb_num}_elbow" type="hinge" axis="-1 0 0" pos="0 0 0" range="-50 50"/>
        <geom type="{body_type}" size="{limb_size}" fromto="0 0 0 0 0 1" rgba="{limb_color}"/>
    """


    with open(f"beginning_of_life.xml", 'a') as file:
        file.write(bod_limb)


    for limb_add in range(1, limb_adds + 1):
        angle_x = random.randint(0, 120)
        angle_y = random.randint(0, 120)
        angle_z = random.randint(0, 120)

        bod_limb = f"""
        <body pos="0 0 1" euler="{angle_x} {angle_y} {angle_z}">
            <joint name="{limb_num}_elbow{limb_add}" type="hinge" axis="-1 0 0" pos="0 0 0" range="-50 50"/>
            <geom type="{body_type}" size="{limb_size}" fromto="0 0 0 0 0 1" rgba="{limb_color}"/>
        """

        with open(f"beginning_of_life.xml", 'a') as file:
            file.write(bod_limb)





    for limb_add in range(1, limb_add + 2):

        body_close = """
        </body>

        """
        with open(f"beginning_of_life.xml", 'a') as file:
            file.write(body_close)



central_body_close = """
</body>

"""
with open(f"beginning_of_life.xml", 'a') as file:
    file.write(central_body_close)







worldbody_close = """
</worldbody>
"""

with open(f"beginning_of_life.xml", 'a') as file:
    file.write(worldbody_close)

####################################
# Actuators

actuator_open = """
<actuator>
"""
with open(f"beginning_of_life.xml", 'a') as file:
    file.write(actuator_open)






for limb_num in range(1, limb_nums + 1):

    bod_limb = f"""
    <motor name="{limb_num}_belly"  gear="40"  joint="{limb_num}_elbow"/>
    """


    with open(f"beginning_of_life.xml", 'a') as file:
        file.write(bod_limb)


    for limb_add in range(1, limb_adds + 1):
        bod_limb = f"""
        <motor name="{limb_num}_belly_{limb_add}"  gear="40"  joint="{limb_num}_elbow{limb_add}"/>
        """


        with open(f"beginning_of_life.xml", 'a') as file:
            file.write(bod_limb)






actuator_close = """
</actuator>
"""
with open(f"beginning_of_life.xml", 'a') as file:
    file.write(actuator_close)

####################################

####################################
close_model = """
</mujoco>
"""

with open(f"beginning_of_life.xml", 'a') as file:
    file.write(close_model)

####################################
    

# # Simulation
# ####################################



model = mujoco.MjModel.from_xml_path('beginning_of_life.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

actuators = model.nu
# move = np.array([10, 10, 10, 10, 10])

move = np.ones(limb_nums*(limb_adds+1)) * 10

print(len(move))


data.ctrl[:actuators] = move



# Run
for i in range(1000):
    if viewer.is_alive:
        if i % 40 == 0:
            # Switch movement direction back and forth every 40 steps
            move *= -1
            
        # Print the timestep
        print(f"Time Step| {i}")


        data.ctrl[:actuators] = move
        mujoco.mj_step(model, data)
        viewer.render()

    else:
        break

viewer.close()










