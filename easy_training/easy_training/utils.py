import numpy as np
import torch
import torch.nn.functional as F
import tf_transformations as tf
    
    
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


def quaternion_to_6d(quat: list) -> list:
    """
    Convert quaternion [x,y,z,w] to Gram-Schmidt 6D rotation representation
    """
    rot_matrix = tf.quaternion_matrix(quat)[:3, :3]

    col1 = rot_matrix[:, 0]
    col2 = rot_matrix[:, 1]

    rot6d = np.concatenate([col1, col2])
    return rot6d.tolist()


def rot6d_to_quaternion(rot6d: list) -> list:
    """
    Convert 6D Gram-Schmidt representation back to quaternion [x,y,z,w]
    """

    a1 = np.array(rot6d[0:3])
    a2 = np.array(rot6d[3:6])

    # Gram-Schmidt
    b1 = a1 / np.linalg.norm(a1)

    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / np.linalg.norm(a2)

    b3 = np.cross(b1, b2)

    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = np.stack([b1, b2, b3], axis=1)

    quat = tf.quaternion_from_matrix(rot_matrix)

    return [quat[0], quat[1], quat[2], quat[3]]

def rot6d_to_matrix(rot6d):
    a1 = rot6d[:3]
    a2 = rot6d[3:]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def do_transform(delta_pose, current_pose):
    """
    delta_pose: [dx, dy, dz, drot_x, drot_y, drot_z]  (axis-angle)
    current_pose: [x, y, z, rot6d(6)]
    """

    # --- Position ---
    delta_pos = np.array(delta_pose[0:3])
    current_pos = np.array(current_pose[0:3])
    new_pos = current_pos + delta_pos

    # --- Current rotation ---
    current_rot6d = np.array(current_pose[3:9])
    current_R = rot6d_to_matrix(current_rot6d)

    # convert to quaternion
    current_T = np.eye(4)
    current_T[:3, :3] = current_R
    current_q = tf.quaternion_from_matrix(current_T)

    # --- Delta rotation (axis-angle) ---
    delta_rotvec = np.array(delta_pose[3:6])
    angle = np.linalg.norm(delta_rotvec)

    if angle < 1e-8:
        delta_q = [0, 0, 0, 1]
    else:
        axis = delta_rotvec / angle
        delta_q = tf.quaternion_about_axis(angle, axis)

    # compose rotations
    new_q = tf.quaternion_multiply(delta_q, current_q)

    # back to rotation matrix
    new_T = tf.quaternion_matrix(new_q)
    new_R = new_T[:3, :3]

    # convert to 6D
    new_rot6d = matrix_to_rot6d(new_R)

    transformed_pose = np.concatenate([new_pos, new_rot6d])

    return transformed_pose.tolist()

def do_reverse_transform(target_pose, current_pose):
    """
    Compute delta_pose that transforms current_pose -> target_pose
    """

    # --- Position ---
    target_pos = np.array(target_pose[0:3])
    current_pos = np.array(current_pose[0:3])
    delta_pos = target_pos - current_pos

    # --- Rotation ---
    target_rot6d = np.array(target_pose[3:9])
    current_rot6d = np.array(current_pose[3:9])

    target_R = rot6d_to_matrix(target_rot6d)
    current_R = rot6d_to_matrix(current_rot6d)

    # relative rotation
    relative_R = target_R @ current_R.T

    relative_T = np.eye(4)
    relative_T[:3, :3] = relative_R

    axis, angle, _ = tf.rotation_from_matrix(relative_T)

    delta_rotvec = np.array(axis) * angle

    delta_pose = np.concatenate([delta_pos, delta_rotvec])

    return delta_pose.tolist()

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
        self.writer.add_scalar("loss/val_loss", losses.get("val_loss", 0.0), self.step)

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
        
        
def test_tf():
    # Test the transformation functions
    current_pose = [0.5, 0.0, 0.2, 1, 0, 0, 0, 1, 0]  # x,y,z + rot6d
    delta_pose = [0.1, 0.0, -0.1, 0.1, 0.2, 0.3]  # dx,dy,dz + drot_x,drot_y,drot_z (axis-angle)

    transformed_pose = do_transform(delta_pose, current_pose)
    print("Transformed Pose:", transformed_pose)
    
    # Now reverse transform
    recovered_delta = do_reverse_transform(transformed_pose, current_pose)
    print("Recovered Delta Pose:", recovered_delta)
    
    
if __name__ == "__main__":
    test_tf()
