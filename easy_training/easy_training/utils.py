import numpy as np
    
    
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
    ori_dist = quaternion_distance(np.array(pose1[3:7]), np.array(pose2[3:7]))
    return pos_dist + ori_dist


def joint_distance(joint1, joint2):
    """Calculate the distance between two joint configurations."""
    return np.linalg.norm(np.array(joint1) - np.array(joint2))

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

from torch.utils.tensorboard import SummaryWriter


class LossVisualizer:

    def __init__(self, log_dir="runs/rl_training", flush_secs=10):
        """
        TensorBoard loss logger

        Parameters
        ----------
        log_dir : str
            TensorBoard log directory
        flush_secs : int
            How often to flush logs to disk
        """

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.step = 0

    def update(self, losses: dict):
        """
        Log loss values to TensorBoard

        Parameters
        ----------
        losses : dict
            {
                "q1_loss": float,
                "q2_loss": float,
                "sac_loss": float,
                "bc_loss": float
            }
        """

        self.step += 1

        # log grouped losses
        self.writer.add_scalar("loss/q1_loss", losses.get("q1_loss", 0.0), self.step)
        self.writer.add_scalar("loss/q2_loss", losses.get("q2_loss", 0.0), self.step)
        self.writer.add_scalar("loss/sac_loss", losses.get("sac_loss", 0.0), self.step)
        self.writer.add_scalar("loss/bc_loss", losses.get("bc_loss", 0.0), self.step)

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
