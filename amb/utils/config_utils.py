"""Tools for loading and updating configs.""" ""
import time
import os
import json
import yaml
from uu import Error


def get_defaults_yaml_args(algo, env, victim=None):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)

    victim_args = {}
    if victim is not None:
        victim_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{victim}.yaml")
        with open(victim_cfg_path, "r", encoding="utf-8") as file:
            victim_args = yaml.load(file, Loader=yaml.FullLoader)

    return algo_args, env_args, victim_args


def get_one_yaml_args(name, type="algo"):
    """Load config file for user-specified algo and env.
    Args:
        name: (str) yaml name.
        type: (str) choose from algo, env
    Returns:
        read_args: (dict) config from yaml.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    cfg_path = os.path.join(base_path, "configs", f"{type}s_cfgs", f"{name}.yaml")


    with open(cfg_path, "r", encoding="utf-8") as file:
        read_args = yaml.load(file, Loader=yaml.FullLoader)

    return read_args

def update_args(unparsed_dict, **kwargs):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(name, dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(name, dict1, dict2[k])
            else:
                if f"{name}.{k}" in dict1:
                    dict2[k] = dict1[f"{name}.{k}"]

    for name in kwargs:
        update_dict(name, unparsed_dict, kwargs[name])
    

def get_task_name(env, env_args):
    """Get task name."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smac_dual":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        if env_args['continuous_actions']:
            task = f"{env_args['scenario']}-continuous"
        else:
            task = f"{env_args['scenario']}-discrete"
    elif env == "gym":
        task = env_args["scenario"]
    elif env == "football":
        task = env_args["env_name"]
    elif env == "dexhands":
        task = env_args["task"]
    elif env == "toy":
        task = "toy"
    return task


def init_dir(env, env_args, algo, exp_name, run_name, seed, logger_path):
    """Init directory for saving results."""
    # check logger_path == nni
    if logger_path == "#nni_dynamic":
        logger_path = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
        results_path = logger_path
    else:
        task = get_task_name(env, env_args)
        hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        # results_path = os.path.join(
        #     logger_path, env, task, run_name, algo, exp_name, '-'.join(['seed-{:0>5}'.format(seed), hms_time])
        # )
        results_path = os.path.join(
            logger_path, env, task, run_name, algo, exp_name, 'seed-{:0>5}'.format(seed)
        )
    print("The experiment path is at:", results_path)
    log_path = os.path.join(results_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writter = SummaryWriter(log_path)
    models_path = os.path.join(results_path, 'models')
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter

# 生成8位随机字符串
def random_str(num):
    import random
    import string
    salt = ''.join(random.sample(string.ascii_letters + string.digits, num))
    return salt

def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), 'w', encoding='utf-8') as out:
        out.write(output)


# 遍历nni_params，如果发现key是 xx.xx.xx 的形式，就把它转换成dict
def convert_nested_dict(params):
    new_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            new_value = convert_nested_dict(value)
        else:
            new_value = value

        if '.' in key:
            print(key)
            keys = key.split('.')
            current_dict = new_params
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[keys[-1]] = new_value
        else:
            new_params[key] = new_value
        
    return new_params

def nni_update_args(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if type(dict1[key]) == type(value):
                if isinstance(dict1[key], dict) and isinstance(value, dict):
                    nni_update_args(dict1[key], value)
                else:
                    dict1[key] = value

def parse_timestep(timesteps, ep_length):
    if timesteps is None:
        return [True for _ in range(1010)]
    if not isinstance(timesteps, str):
        return timesteps
    timesteps = timesteps.strip().replace(" ", "").split(",")
    parsed_steps = []
    for tp in timesteps:
        if "-" in tp:
            start, end = tp.split("-")
            parsed_steps.extend(list(range(int(start), int(end) + 1)))
        else:
            parsed_steps.append(int(tp))
    steps_bool = [False for _ in range(ep_length)]
    for step in parsed_steps:
        steps_bool[step] = True
    return steps_bool

