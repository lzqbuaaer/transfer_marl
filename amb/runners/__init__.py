def get_single_runner(run, algo):
    if run == "single":
        if algo == "mappo":
            from amb.runners.single.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.single.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner
    if run == "perturbation":
        if algo == "mappo":
            from amb.runners.perturbation.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.perturbation.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner
    if run == "traitor":
        if algo == "mappo":
            from amb.runners.traitor.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.traitor.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner

def get_dual_runner(run, algo):
    if run == "dual":
        if algo == "mappo":
            from amb.runners.dual.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.dual.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner
    if run == "perturbation":
        if algo == "mappo":
            from amb.runners.perturbation.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.perturbation.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner
    if run == "traitor":
        if algo == "mappo":
            from amb.runners.dual_traitor.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.dual_traitor.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner