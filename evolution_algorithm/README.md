#Evolution Unleashed: Creating Life from Code - Witness the Astonishing Journey of AI Creatures!


This project was constructed by Arman Moussavi for Artificial Life, instructed by Dr. Sam Kriegman.

Here, an algorithm for evolutionary selection is demonstrated.
Creatures are built, tested, and selected using the “code begin.py” (see description). This code tests a specified number of creatures in a single population and then tests multiple generations of populations. 

The creatures are built initially with a single cube as a central body. A random selection of the number and location of limbs is conducted to place limbs randomly on the horizontal faces of the central body. Limbs hold actuators with a constant open and closing action to stimulate the movement of the creature. A selection criteria is employed to retain the creature that shows the highest fitness for the task at hand. The task here is to move as far as possible in the positive X-direction. There is a penalty for excess Y-direction displacement. The best creature from a population is saved and then mutated as the next generation of creatures. Mutations signify the addition of limbs to random locations on the preexisting best creature of the last generation. 

For each plateau experienced in the process of life plot previously displayed, the simulation is shown now, so one may see how the creature has evolved.

The creatures of the new generation are evaluated for their fitness and the best creature is selected. Finally, the best creature from the current generation is compared with the best creature from the previous generation, and the better creature is selected as a host for the next generation.


To execute this code, please download all necessary packages. At the bottom of “begin.py”, under settings, the user may change the global_seed for random interactions of the simulation. The simulation settings allow the user to specify the number of steps in the testing simulation of each creature, the timestep used, and the environmental conditions. Under lineage settings, the user may specify the number of generations over which to evaluate the evolution, as well as the population size of each generation. Finally, the appearance settings allow the user to specify the maximum number of limbs as well as limb mutations to be randomly selected throughout the process of evolution.


The code will output “life.xml”, the body file for Mujoco; “execution_time.txt”, the time the code took to execute; “blueprints_to_life.txt”, the creature coordinates for future reference; “process_of_life_plot.png”, fitness increase across generations; “generation_traj.png”, x-y plane trajectory of the selected creature to view.


Additionally, the code “read_blueprints.py” allows the user to read in blueprints previously made by “begin.py” and visualize selected creatures to avoid the need to re-run the full simulation test procedure.


All coding logic was manually derived. Technical implementation of the logic as well as debugging and cleaning was aided by pulling snips of code from various sources as well as help from ChatGPT. 
Code is written in the open-source language Python (Van Rossum, G., & Drake Jr, F. L. (1995). Python reference manual. Centrum voor Wiskunde en Informatica Amsterdam). Simulations were run using the physics engine Mujoco (Todorov, E., Erez, T. & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control.. IROS (p./pp. 5026-5033), : IEEE. ISBN: 978-1-4673-1737-5). 


