# datapoints experiment
# creates step_map
# tells listener.py to run

def datapoint_experiment(node_num_desired):
    jobs = [dict(seed=i, algorithm="random", info="pending") for i in range(node_num_desired)]