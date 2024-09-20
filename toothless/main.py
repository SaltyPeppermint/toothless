from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
from torch_geometric.data import Data

import eggshell

import symbols
from rl_env import SketchEnv
from data import seed_pairs
from ppo import PPO


device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


DATA_PATH = Path(
    "data/with_baseline/5k_dataset_2024-09-04_10:02:20-b3e59ffa-b9f7-4807-80e7-3f20ecc18c8f"
)


def make_env(
    data: list[(Data, Data)],
    symbol_list: list[symbols.Symbol],
    symbol_table: dict[str, symbols.Symbol],
) -> gym.Env:
    data = seed_data

    max_size = 30
    min_size = 10
    max_ratio_sketch = 0.2

    env = SketchEnv(
        flat_term_pairs=data,
        max_size=max_size,
        min_size=min_size,
        max_sketch_ratio=max_ratio_sketch,
        symbol_table=symbol_table,
        actions=halide_sketch_symbols,
        node_tensor_len=node_tensor_len,
        typechecker=eggshell.halide.typecheck_sketch,
        render_mode=None,
    )
    return env


################################### Training ###################################
def train(device: torch.DeviceObjType, env: gym.Env):
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(
        3e6
    )  # break training loop if timeteps > max_training_timesteps

    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 1024  # set random seed if required (0 = no random seed)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = (
        0  #### change this to prevent overwriting weights in same env_name folder
    )
    checkpoint_folder = Path("checkpoints")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    run_num_pretrained = checkpoint_folder.glob("*")

    checkpoint_path = checkpoint_folder / Path(
        f"PPO_{random_seed}_{run_num_pretrained}.pth"
    )

    ###################### logging ######################
    logger = SummaryWriter()

    ############# print save location ###################
    print("===========================================================================")
    print(f"save checkpoint path : {checkpoint_path}")

    ############# print all hyperparameters #############
    print("---------------------------------------------------------------------------")
    print(f"max training timesteps : {max_training_timesteps}")
    print(f"max timesteps per episode : {max_ep_len}")
    print(f"model saving frequency : {save_model_freq} timesteps")
    # print("log frequency : " + str(log_freq) + " timesteps")
    print("---------------------------------------------------------------------------")
    print(f"state space dimension : {env.observation_space["sketch"].node_space.shape}")
    print(f"action space dimension : {env.action_space.n}")
    print("---------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")
    print("---------------------------------------------------------------------------")
    print(f"PPO update frequency : {update_timestep} timesteps")
    print(f"PPO K epochs : {K_epochs}")
    print(f"PPO epsilon clip : {eps_clip}")
    print(f"discount factor (gamma) : {gamma}")
    print("---------------------------------------------------------------------------")
    print(f"optimizer learning rate actor : {lr_actor}")
    print(f"optimizer learning rate critic : {lr_critic}")
    if random_seed:
        print(
            "---------------------------------------------------------------------------"
        )
        print(f"setting random seed to {random_seed}")
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("===========================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        device,
        env,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("===========================================================================")

    time_step = 0
    i_episode = 0
    episode_len = 0

    # training loop
    while time_step <= max_training_timesteps:
        observation, info = env.reset()

        current_ep_reward = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(observation)
            observation, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            episode_len += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # save model weights
            if time_step % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

            # break; if the episode is over
            if done:
                break

        logger.add_scalar("Episode Length", episode_len)
        logger.add_scalar("Episode Reward", current_ep_reward, time_step)

        episode_len = 0

        i_episode += 1

    env.close()

    # print total training time
    print("===========================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("===========================================================================")


if __name__ == "__main__":
    halide_sketch_symbols = symbols.halide_symbols(10, 0)
    symbol_table = symbols.symbol_table(
        symbols.add_partial_symbols(halide_sketch_symbols)
    )

    seed_data = seed_pairs(DATA_PATH, symbol_table)
    print(f"Seed dataset: {len(seed_data)}")

    node_tensor_len = len(halide_sketch_symbols) + 5

    env = make_env(seed_data, halide_sketch_symbols, symbol_table)
    train(device, env)
