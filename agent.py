import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, use_dml=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_episode = 0
        self.curr_step = 0
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e4  # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_dml = use_dml
        self.use_cuda = torch.cuda.is_available()
        self.device = None
        self.max_dict = dict()
        for world in range(1, 9):
            for stage in range(1, 5):
                self.max_dict[(world, stage)] = {
                    "x_pos": 0,
                    "score": 0,
                    "time_left": 0,
                    "coins": 0,
                }
        self.max_world = 1
        self.max_stage = 1

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_dml:
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
        else:
            self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.net = self.net.to(self.device)
        if checkpoint:
            self.load(checkpoint)

        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)  # type: ignore
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.00025, momentum=0.9
        )
        self.loss_fn = torch.nn.SmoothL1Loss()

    def visualize_weights(self):
        """
        Extract the weights of the first convolutional layer and reshape them to match the input dimensions.
        """
        with torch.no_grad():
            weights = self.net.online[0].weight.cpu().clone()
            # Assuming input is 1 channel, output is 32 channels, kernel size is 8x8
            # weights shape is (32, 1, 8, 8)
            # Reshape weights to (32, 8, 8)
            weights = weights.squeeze(1)
            # Normalize weights to [0, 1]
            weights_min = weights.min()
            weights_max = weights.max()
            weights = (weights - weights_min) / (weights_max - weights_min)
            # Resize weights to 84x84
            weights = torch.nn.functional.interpolate(
                weights.unsqueeze(1),
                size=(84, 84),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            return weights

    def reset_max(self, world, stage):
        self.max_dict[(world, stage)] = {
            "x_pos": 0,
            "score": 0,
            "time_left": 0,
            "coins": 0,
        }
        self.max_world = 1
        self.max_stage = 1

    def reset_max_all(self):
        self.max_dict = dict()
        for world in range(1, 9):
            for stage in range(1, 5):
                self.reset_max(world, stage)

    def update_max(self, info, win=False):
        world = info["world"]
        stage = info["stage"]
        if info["x_pos"] > self.max_dict[(world, stage)]["x_pos"]:
            self.max_dict[(world, stage)]["x_pos"] = info["x_pos"]
        if info["score"] > self.max_dict[(world, stage)]["score"]:
            self.max_dict[(world, stage)]["score"] = info["score"]
        if win:
            if "time_left" not in self.max_dict[(world, stage)]:
                self.max_dict[(world, stage)]["time_left"] = 0

            if (
                "time_left" in info
                and info["time_left"] > self.max_dict[(world, stage)]["time_left"]
            ):
                self.max_dict[(world, stage)]["time_left"] = info["time_left"]
        if info["coins"] > self.max_dict[(world, stage)]["coins"]:
            self.max_dict[(world, stage)]["coins"] = info["coins"]
        if world > self.max_world:
            self.max_world = world
            self.max_stage = stage
        if world == self.max_world and stage > self.max_stage:
            self.max_stage = stage

    def get_max(self, world, stage):
        return self.max_dict[(world, stage)]

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(np.array(state)).to(self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()  # type: ignore

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """

        while True:
            try:
                state_tensor = torch.FloatTensor(np.array(state)).to(self.device)
                next_state_tensor = torch.FloatTensor(np.array(next_state)).to(
                    self.device
                )
                action_tensor = torch.LongTensor(np.array([action])).to(self.device)
                reward_tensor = torch.DoubleTensor(np.array([reward])).to(self.device)
                done_tensor = torch.BoolTensor(np.array([done])).to(self.device)
                break
            except Exception as e:
                print(e)
                # print("Out of memory, reducing memory size")
                _ = [self.memory.popleft() for _ in range(1000)]

        self.memory.append(
            (state_tensor, next_state_tensor, action_tensor, reward_tensor, done_tensor)
        )

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)  # type: ignore
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        while True:
            try:
                if self.curr_step % self.sync_every == 0:
                    self.sync_Q_target()

                if self.curr_step % self.save_every == 0:
                    self.save()

                if self.curr_step < self.burnin:
                    return None, None

                if self.curr_step % self.learn_every != 0:
                    return None, None

                if len(self.memory) < self.batch_size:
                    return None, None

                # Sample from memory
                state, next_state, action, reward, done = self.recall()

                # Get TD Estimate
                td_est = self.td_estimate(state, action)

                # Get TD Target
                td_tgt = self.td_target(reward, next_state, done)

                # Backpropagate loss through Q_online
                loss = self.update_Q_online(td_est, td_tgt)
                break
            except Exception as e:
                _ = [self.memory.popleft() for _ in range(1000)]
        return (td_est.mean().item(), loss)

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                max_dict=self.max_dict,
                max_world=self.max_world,
                max_stage=self.max_stage,
                step=self.curr_step,
                episodes=self.curr_episode,
            ),
            save_path,
        )

        # Limit to 10 saved files
        saved_files = sorted(
            Path(self.save_dir).iterdir(), key=lambda f: f.stat().st_mtime
        )
        if len(saved_files) > 10:
            file_to_delete = saved_files[0]
            file_to_delete.unlink()

        # print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=("cuda" if self.use_cuda else "cpu"))
        exploration_rate = ckp.get("exploration_rate")
        state_dict = ckp.get("model")

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(self.device)
        self.exploration_rate = exploration_rate
        self.max_dict = ckp.get("max_dict")
        self.max_world = ckp.get("max_world")
        self.max_stage = ckp.get("max_stage")
        self.curr_step = ckp.get("step")
        self.curr_episode = ckp.get("episodes")
