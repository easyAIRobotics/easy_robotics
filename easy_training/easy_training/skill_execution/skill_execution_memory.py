import torch
import numpy as np
import pickle
import gzip
from easy_training.agent_interfaces import ReplayBuffer


class SkillExecutionReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity,
        device="cuda",
        max_episode_len=100,
        seq_len=16,
        use_float16=False,
    ):
        super().__init__(capacity, device)

        self.seq_len = seq_len
        self.max_episode_len = max_episode_len
        self.use_float16 = use_float16

        self.episodes = []
        self.current_episode = self._init_episode()

        self.total_steps = 0

    # ======================================================
    def _dtype(self):
        return np.float16 if self.use_float16 else np.float32

    # ======================================================
    def _init_episode(self):
        return {
            "images": [],
            "target_images": [],
            "skills": [],
            "robot_states": [],
            "j_actions": [],
            "e_actions": [],
            "rewards": [],
            "dones": [],
            "next_images": [],
            "next_target_images": [],
            "next_skills": [],
            "next_robot_states": [],
        }

    # ======================================================
    def add(
        self,
        image,
        target_image,
        skill,
        robot_state,
        j_action,
        e_action,
        reward,
        next_image,
        next_target_image,
        next_skill,
        next_robot_state,
        done
    ):
        if done and len(self.current_episode["images"]) == 0:
            return

        ep = self.current_episode

        ep["images"].append(image)
        ep["target_images"].append(target_image)
        ep["skills"].append(skill)
        ep["robot_states"].append(robot_state)

        ep["j_actions"].append(j_action)
        ep["e_actions"].append(e_action)
        ep["rewards"].append(reward)
        ep["dones"].append(done)

        ep["next_images"].append(next_image)
        ep["next_target_images"].append(next_target_image)
        ep["next_skills"].append(next_skill)
        ep["next_robot_states"].append(next_robot_state)

        self.total_steps += 1

        if done or len(ep["images"]) >= self.max_episode_len:
            self._finalize_episode()

    # ======================================================
    def _finalize_episode(self):
        ep = self.current_episode
        T = len(ep["images"])
        if T == 0:
            return

        dtype = self._dtype()

        for k in ep:
            ep[k] = np.asarray(ep[k], dtype=dtype)

        self.episodes.append(ep)

        # capacity control
        while self.total_steps > self.capacity and len(self.episodes) > 0:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed["images"])

        self.current_episode = self._init_episode()

    # ======================================================
    def sample(self, batch_size):
        if len(self.episodes) == 0:
            return None

        samples = []
        attempts = 0
        max_attempts = batch_size * 20

        while len(samples) < batch_size and attempts < max_attempts:
            attempts += 1

            ep = self.episodes[np.random.randint(len(self.episodes))]
            T = len(ep["images"])

            if T < self.seq_len:
                continue

            start = np.random.randint(0, T - self.seq_len + 1)
            end = start + self.seq_len

            def get(k):
                return ep[k][start:end]

            samples.append({
                "images": get("images"),
                "target_images": get("target_images"),
                "skills": get("skills"),
                "robot_states": get("robot_states"),
                "next_images": get("next_images"),
                "next_target_images": get("next_target_images"),
                "next_skills": get("next_skills"),
                "next_robot_states": get("next_robot_states"),
                "j_actions": get("j_actions"),
                "e_actions": get("e_actions"),
                "rewards": get("rewards"),
                "dones": get("dones"),
            })

        if len(samples) == 0:
            return None

        def stack(key):
            return np.stack([s[key] for s in samples], axis=0)

        images = stack("images")
        target_images = stack("target_images")
        skills = stack("skills")
        robot_states = stack("robot_states")

        next_images = stack("next_images")
        next_target_images = stack("next_target_images")
        next_skills = stack("next_skills")
        next_robot_states = stack("next_robot_states")

        j_actions = stack("j_actions")
        e_actions = stack("e_actions")
        rewards = stack("rewards")
        dones = stack("dones")

        def to_torch(x):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)

        images = to_torch(images).permute(0, 1, 4, 2, 3)
        target_images = to_torch(target_images).permute(0, 1, 4, 2, 3)
        next_images = to_torch(next_images).permute(0, 1, 4, 2, 3)
        next_target_images = to_torch(next_target_images).permute(0, 1, 4, 2, 3)

        return (
            (images, target_images, to_torch(skills), to_torch(robot_states)),
            to_torch(j_actions),
            to_torch(e_actions),
            to_torch(rewards),
            to_torch(dones),
            (next_images, next_target_images, to_torch(next_skills), to_torch(next_robot_states)),
        )

    # ======================================================
    # SAVE / LOAD
    # ======================================================
    def save_to_disk(self, filepath):
        data = {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
        }

        with gzip.open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_disk(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = pickle.load(f)

        self.episodes = data["episodes"]
        self.total_steps = data["total_steps"]
        self.current_episode = self._init_episode()

    # ======================================================
    def size(self):
        return len(self.episodes)