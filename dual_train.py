import os
import argparse
import json
from pprint import pprint
from amb.utils.config_utils import get_one_yaml_args, update_args, parse_timestep
from amb.utils.env_utils import get_manual_env_prior

def main():
    """Main function."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--angel",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix",
        ],
        help="Algorithm name. Choose from: maddpg, mappo, igs.",
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix",
        ],
        help="Algorithm name. Choose from: maddpg, mappo, igs.",
    )
    parser.add_argument(
        "--demon",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix"
        ],
        help="Victim algorithm name. Choose from: maddpg, mappo, qmix.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="dual",
        choices=[
            "dual",
            "perturbation",
            "traitor",
        ],
        help="Runner pipeline name. Choose from: dual, perturbation, traitor.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="smac_dual",
        choices=[
            "smac_dual",
        ],
        help="Environment name. Choose from: smac_dual.",
    )
    parser.add_argument("--exp_name", type=str, default="installtest", help="Experiment name.")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_angel",
        type=str,
        default="",
        help="If set, load existing angel config file and checkpoint file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_demon",
        type=str,
        default="",
        help="If set, load existing demon config file and checkpoint file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_victim",
        type=str,
        default="",
        help="If set, load existing angel config file and checkpoint file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding='utf-8') as file:
            all_config = json.load(file)
        args["angel"] = all_config["main_args"]["angel"]
        args["demon"] = all_config["main_args"]["demon"]
        args["run"] = all_config["main_args"]["run"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]

        angel_args = all_config["algo_args"]["angel"]
        demon_args = all_config["algo_args"]["demon"]
        victim_args = all_config["algo_args"]["victim"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        env_args = get_one_yaml_args(args["env"], type="env")

        if args["load_angel"] != "":
            with open(os.path.join(args["load_angel"], "config.json"), encoding='utf-8') as file:
                angel_config = json.load(file)

            if "algo" in angel_config["main_args"]:
                args["angel"] = angel_config["main_args"]["algo"]
                angel_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_angel"], "models")
                angel_args = angel_config["algo_args"]["train"]
            else:
                args["angel"] = angel_config["main_args"]["angel"]
                angel_config["algo_args"]["angel"]["model_dir"] = os.path.join(args["load_angel"], "models", "angel")
                angel_args = angel_config["algo_args"]["angel"]

                args["env"] = angel_config["main_args"]["env"]
                env_args = angel_config["env_args"]
        else:
            if args["run"] == "dual":
                angel_args = get_one_yaml_args(args["angel"])
            elif args["run"] == "perturbation" or args["run"] == "traitor":
                angel_args = get_one_yaml_args(args["angel"] + "_traitor")

        victim_args = {}
        if args["load_victim"] != "":
            with open(os.path.join(args["load_victim"], "config.json"), encoding='utf-8') as file:
                victim_config = json.load(file)
            if "algo" in victim_config["main_args"]:
                args["victim"] = victim_config["main_args"]["algo"]
                victim_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_victim"], "models")
                victim_args = victim_config["algo_args"]["train"]
            else:
                raise NotImplementedError
                # args["demon"] = victim_config["main_args"]["demon"]
                # victim_config["algo_args"]["demon"]["model_dir"] = os.path.join(args["load_victim"], "models", "angel")
                # victim_args = victim_config["algo_args"]["demon"]
        elif args["run"] == "perturbation" or args["run"] == "traitor":
            victim_args = get_one_yaml_args(args["victim"])

        if args["load_demon"] != "":
            with open(os.path.join(args["load_demon"], "config.json"), encoding='utf-8') as file:
                demon_config = json.load(file)
            if "algo" in demon_config["main_args"]:
                args["demon"] = demon_config["main_args"]["algo"]
                demon_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_demon"], "models")
                demon_args = demon_config["algo_args"]["train"]
            else:
                args["demon"] = demon_config["main_args"]["demon"]
                demon_config["algo_args"]["demon"]["model_dir"] = os.path.join(args["load_demon"], "models", "demon")
                demon_args = demon_config["algo_args"]["demon"]
        else:
            demon_args = get_one_yaml_args(args["demon"])
            
    update_args(unparsed_dict, angel=angel_args, env=env_args, demon=demon_args, victim=victim_args)  # update args from command line
    algo_args = {"angel": angel_args, "demon": demon_args, "victim": victim_args}

    if "perturb_timesteps" in algo_args["angel"]:
        algo_args["angel"]["perturb_timesteps"] = parse_timestep(algo_args["angel"]["perturb_timesteps"], algo_args["angel"]["episode_length"])

    pprint([args, algo_args, env_args])
    if "use_manual_env_prior" in algo_args["angel"] and algo_args["angel"]["use_manual_env_prior"]:
        algo_args["angel"]["manual_env_prior"] = get_manual_env_prior(args["env"], env_args)

    # start training
    from amb.runners import get_dual_runner
    runner = get_dual_runner(args["run"], args["angel"])(args, algo_args, env_args)
    if algo_args["angel"]['use_render']:  # render, not train
        runner.render()
    else:
        runner.run()
    runner.close()


if __name__ == "__main__":
    main()
