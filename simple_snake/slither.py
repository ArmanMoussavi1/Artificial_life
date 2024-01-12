import mujoco
import mujoco_viewer
import numpy as np



model = mujoco.MjModel.from_xml_path('snake.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

actuators = model.nu
move = np.array([10, 10, 10, 10, 10])
data.ctrl[:actuators] = move



# Run
for i in range(10000):
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










