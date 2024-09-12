# Transferrable Adversarial Attack
> Based on [AMB: Adversarial MARL Benchmark]

代码相较于原本AMB框架主要添加以下四部分内容：
1. 合作维度对齐(3m ==> 4m_vs_3m)
   将训练后合作智能体模型参数迁移至友军+1对抗场景时做出的维度错位对齐调整
2. 对抗维度对齐(4m_vs_3m/6m ==> 9m_vs_8m)
   训练并测试对抗智能体能力时做的尾部添零维度对齐
3. 训练神经网络模块
   环境编码器与网络的添加
4. 训练流程
   分轮次、分模块训练

以3m/8m两张地图上训练adv-agent，5m_vs_6m上测试为例给出指令：
总共使用3m, 8m, 5m_vs_6m, 4m_vs_3m, 9m_vs_8m, 6m六张地图；
六张地图原本obs/state/action维度信息分别为：
| map      | obs  | state | action |
| -------- | ---- | ----- | ------ |
| 3m       | 64   | 81    | 9      |
| 8m       | 204  | 251   | 14     |
| 5m_vs_6m | 124  | 156   | 12     |
| 4m_vs_3m | 79   | 99    | 9      |
| 9m_vs_8m | 224  | 274   | 14     |
| 6m       | 142  | 177   | 12     |

使用步骤：
1. 训练合作智能体
   ```
   python -u single_train.py --env smac --exp_name 3m-victim --algo mappo --run single --env.map_name 3m --algo.log_dir ./results/smac/[1]-victims
   python -u single_train.py --env smac --exp_name 8m-victim --algo mappo --run single --env.map_name 8m --algo.log_dir ./results/smac/[1]-victims
   python -u single_train.py --env smac --exp_name 5m_vs_6m-victim --algo mappo --run single --env.map_name 5m_vs_6m --algo.log_dir ./results/smac/[1]-victims
   ```

2. MAPPO/transfer训练本方+1的对抗智能体
   注1：在两张地图上分别进行1,000,000步训练，共三组；首组动态模块第一层开grad_on并更新，后续训练组别时关掉；
   注2：adv_path_i/final_adv_path与map-victim-path均为路径代号，使用时需要改为实际路径
   ```
   python -u single_train.py --env smac --exp_name adv-agent --algo mappo --run traitor --env.map_name 4m_vs_3m --algo.num_env_steps 1000000 --algo.log_dir adv_path_1 --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 3m-victim-path --algo.static_env_net True
   
   python -u single_train.py --env smac --exp_name adv-agent --algo mappo --run traitor --env.map_name 9m_vs_8m --algo.num_env_steps 1000000 --algo.log_dir adv_path_2 --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 8m-victim-path --algo.model_dir adv_path_1 --algo.static_env_net True 
   
   python -u single_train.py --env smac --exp_name adv-agent --algo mappo --run traitor --env.map_name 4m_vs_3m --algo.num_env_steps 1000000 --algo.log_dir adv_path_3 --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 3m-victim-path --algo.model_dir adv_path_2 --algo.static_env_net True --algo.obs_feat_layer_grad_on False
   
   python -u single_train.py --env smac --exp_name adv-agent --algo mappo --run traitor --env.map_name 9m_vs_8m --algo.num_env_steps 1000000 --algo.log_dir adv_path_4 --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 8m-victim-path --algo.model_dir adv_path_3 --algo.static_env_net True --algo.obs_feat_layer_grad_on False
   
   python -u single_train.py --env smac --exp_name adv-agent --algo mappo --run traitor --env.map_name 4m_vs_3m --algo.num_env_steps 1000000 --algo.log_dir adv_path_5 --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 3m-victim-path --algo.model_dir adv_path_4 --algo.static_env_net True --algo.obs_feat_layer_grad_on False
   
   python -u single_train.py --env smac --exp_name adv-agent --algo mappo --run traitor --env.map_name 9m_vs_8m --algo.num_env_steps 1000000 --algo.log_dir final_adv_path --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 8m-victim-path --algo.model_dir adv_path_5 --algo.static_env_net True --algo.obs_feat_layer_grad_on False
   ```

3. +1合作场景下的内鬼测试
   ```
   python -u single_train.py --env smac --exp_name adv-agent-eval --algo.use_render True --algo mappo --run traitor --env.map_name 6m  --algo.log_dir adv_eval_path --algo.obs_state_align True --algo.action_space_align True --algo.adv_agent_ids [0] --load_victim 5m_vs_6m-victim-path --algo.model_dir final_adv_path
   ```
