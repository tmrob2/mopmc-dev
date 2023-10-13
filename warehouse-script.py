import sys
import random

# Check the number of command-line arguments
num_args = len(sys.argv)

fname = sys.argv[1]                # The first argument is the filename
warehouse_size = int(sys.argv[2])  # Second argument is the size of the warehouse
num_agents = int(sys.argv[3])      # Third argument is the number of agents
num_tasks = int(sys.argv[4])       # Forth argument is the number of tasks

# Get some initial positions for the agents (up to the number of agents) in the warehouse

# construct some initial points for the agents.
# place the agents around the edge of the warehouse
# divide the number of agents in two
agents_along_x = num_agents // 2
agent_positions = [(i, 0) for i in range(agents_along_x)]
agents_along_y = num_agents - agents_along_x
yAdditions = [(0, 2 + i) for i in range(agents_along_y)]
agent_positions.extend(yAdditions)

# choose some random square which are not around the outside of the warehouse
possibleTaskPositions = [(i, j) for i in range(1, warehouse_size - 2)
                         for j in range(1, warehouse_size - 2)]

# now randomly select num_tasks from the warehouse positions
print(possibleTaskPositions)
print(agent_positions)
taskObjects = random.sample(possibleTaskPositions, num_tasks)

with open(fname + '.nm', 'w') as file:
    file.write('mdp\n\n')

    # Set the constants
    file.write('//constants\n')
    file.write(f'const int n = {warehouse_size};\n')
    file.write(f'const int numAgents = {num_agents};\n')
    file.write(f'const int numTasks = {num_tasks};\n')
    for i, (agentX, agentY) in enumerate(agent_positions):
        file.write(f'const int initX{i} = {agentX};\n')
        file.write(f'const int initY{i} = {agentY};\n')
    file.write("\n")
    file.write(f'formula left = max(0, x - 1);\n')
    file.write(f'formula right = min(n - 1, x + 1);\n')
    file.write(f'formula up = min(n - 1, y + 1);\n')
    file.write(f'formula down = max(0, y - 1);\n')
    file.write("\n")
    agentInit = "formula agentInit = "
    for i in range(len(agent_positions)):
        file.write(f'formula init{i} = x = initX{i} & y = initY{i};\n')
        if i == len(agent_positions) - 1:
            agentInit += f"init{i}"
        else:
            agentInit += f"init{i} | "
    agentInit += ";\n"
    file.write(agentInit)
    file.write("\n")

    for i in range(num_tasks):
        file.write(f"formula task{i}Obj = x = {taskObjects[i][0]} & y = {taskObjects[i][1]};\n")

    file.write("formula taskIdle = t = 0 | t = 3;\n")
    file.write("formula taskComplete = t = 3;\n")
    file.write("formula missionComplete = t = 3 & active_task = 1;\n")

    # module start
    file.write("module warehouse\n")
    file.write("\tt: [0..3];\n")
    file.write("\tactive_agent: [0..numAgents - 1] init 0;\n")
    file.write("\tactive_task: [0..numTasks - 1] init 0;\n")
    file.write(f"\tx: [0..n] init {agent_positions[0][0]};\n")
    file.write(f"\ty: [0..n] init {agent_positions[0][1]};\n")
    file.write("\n")
    # Programming the switch transitions
    file.write("\t// Switch transitions handing the current task over to the next agent\n")
    for agent in range(num_agents - 1):
        file.write(f"\t[] (agentInit & t = 0 & active_agent = {agent}) -> "
            f"1: (active_agent' = {agent + 1}) & (x' = {agent_positions[agent + 1][0]}) & (y' = {agent_positions[agent + 1][1]});\n")
    file.write("\t//if the current task has been completed then a new task is activated\n")
    file.write("\t[] (agentInit & taskComplete & active_task < numTasks - 1) ->"
               "1: (active_task' = active_task + 1) & (t' = 0) & (active_agent' = 0);\n")
    file.write("\n")
    file.write("\t[] (x >= 0 & x <= n - 1 & y >= 0 & y <= n - 1 & t > 0) -> 1 : (x' = right);\n")
    file.write("\t[] (x >= 0 & x <= n - 1 & y >= 0 & y <= n - 1 & t > 0) -> 1 : (x' = left);\n")
    file.write("\t[] (x >= 0 & x <= n - 1 & y >= 0 & y <= n - 1 & t > 0) -> 1 : (y' = up);\n")
    file.write("\t[] (x >= 0 & x <= n - 1 & y >= 0 & y <= n - 1 & t > 0) -> 1 : (y' = down);\n")
    file.write("\t//Task Components\n")
    file.write("\t[] (agentInit & t = 0) -> 1 : (t' = 1) ;\n")
    for i in range(num_tasks):
        file.write(f"\t[] (task{i}Obj & t = 1 & active_task = {i}) -> 1 : (t' = 2);\n")
    file.write("\t[] (x = 0 & y = 0 & t = 2) -> 1 : (t' = 3);\n")
    file.write("endmodule\n")

    file.write("\n")
    file.write(f'label "finish" = t=2 & active_task = {num_tasks - 1};\n')

    for k in range(num_tasks):
        file.write("\n")
        file.write(f'rewards "movement_cost{k + 1}"\n')
        file.write(f"\t!missionComplete & active_task = {k} : 1;\n")
        file.write("endrewards\n")

    for i in range(num_tasks):
        file.write("\n")
        file.write(f'rewards "task_prob{i + 1}"\n')
        file.write(f"\t!missionComplete & active_task = {i} : 1;\n")
        file.write("endrewards\n")

    file.write("\n")
    file.write(f"rewards \"total\"\n")
    file.write(f"\t!missionComplete : 1; \n")
    file.write("endrewards\n")



