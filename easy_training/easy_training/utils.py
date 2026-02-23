import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, data):
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    
def quaternion_distance(q1, q2):
    """Calculate the distance between two quaternions."""
    cos_t = np.dot(q1, q2)
    if cos_t < 0.0:
        q2 = -q2
        cos_t = -cos_t
        
    return 2 * np.arccos(np.clip(cos_t, -1.0, 1.0))


def transform_distance(pose1, pose2):
    """Calculate a distance metric between two poses (position + orientation)."""
    pos_dist = np.linalg.norm(np.array(pose1[0:3]) - np.array(pose2[0:3]))
    ori_dist = quaternion_distance(pose1[3:7], pose2[3:7])
    return pos_dist + ori_dist


"""
Image pre processing utilities
"""
def depth_msg_to_numpy(msg):
    depth = np.frombuffer(
        msg.data,
        dtype=np.float32
    ).reshape((msg.height, msg.width))
    return depth

def median_downsample(img, k):
    h, w = img.shape
    h2, w2 = h // k, w // k

    img = img[:h2*k, :w2*k]
    img = img.reshape(h2, k, w2, k)

    return np.median(img, axis=(1, 3))