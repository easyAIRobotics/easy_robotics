from easy_training.task_planning.task_planning_cfg import *
from rclpy.node import Node
from easy_training.agent_interfaces import ActionInterface, StateInterface, AgentMode
from easy_training.utils import *

from easy_interfaces.msg import BoundingBox
from easy_interfaces.srv import SetString
from std_msgs.msg import Float64MultiArray


class TaskPlanningActionInterface(ActionInterface):
    def __init__(self, node: Node):
        super().__init__(node)
        self.skill_vector = np.zeros(3, dtype=np.float32)
        self.frequency = 0.5
        self.bbox_publisher = self._node.create_publisher(
            BoundingBox,
            "selected_box",
            1
        )
        
        self.skill_vector_publisher = self._node.create_publisher(
            Float64MultiArray,
            "selected_skill_vector",
            1
        )
        
        self.skill_execution_mode_client = self._node.create_client(
            SetString,
            "skill_execution/set_mode"
        )
        
        self._last_selected_bbox_center = None
        
    def set_action(self, action):
        super().set_action(action)
        skill_vec_msg = Float64MultiArray()
        print(f"Setting action to: {action}, skill vector: {self.skill_vector}", flush=True)
        self.skill_vector = SKILL_VOCAB.get(action, np.zeros(3, dtype=np.float32))
        skill_vec_msg.data = self.skill_vector.tolist()
        self.skill_vector_publisher.publish(skill_vec_msg)
        
    def perform(self, act_vec: list, state_interface: StateInterface) -> dict:
        transition = {
            "rgb_image": state_interface.get_rgb_image(),
            "bbox_list": state_interface.get_bbox_list(),
            "class_list": state_interface.get_class_list(),
            "robot_state": state_interface.get_robot_state(),
        }
        
        reward = 0.0
        if act_vec:
            bbox, class_embedded = self.select_bbox(act_vec[:20], transition["bbox_list"], transition["class_list"])
            self.send_bbox(bbox, class_embedded)
            self.send_skill(act_vec[20:23])
            self.skill_vector = act_vec[20:23]
            self.skill_execution_mode_client.call_async(SetString.Request(data="training/exec"))
            self.wait_for_next_state()
            taken_action = act_vec
            
        else:
            skill_vec = self.skill_vector
            selected_bbox_center = state_interface.get_selected_bbox_center()
            id = self._find_bbox_id_by_center(selected_bbox_center, transition["bbox_list"])
            prob_vec = [0.0] * NUM_HEADS
            prob_vec[id] = 1.0
            taken_action = np.concatenate([prob_vec, skill_vec])
            self.skill_execution_mode_client.call_async(SetString.Request(data="training/exec"))
            self.wait_for_next_state()
            print(f"Selected bbox id: {id}, center: {selected_bbox_center}, skill vector: {skill_vec}", flush=True)
        
        reward = state_interface.get_reward()
        act_done = state_interface.get_done()    
        transition.update({
            "done": act_done,
            "action": taken_action,
        })
        
        return reward, transition
    
    def _find_bbox_id_by_center(self, center, bbox_list):
        if self._last_selected_bbox_center is not None and np.linalg.norm(np.array(center) - np.array(self._last_selected_bbox_center)) < 1e-3:
            return 0
        
        # New selection, update last selected center
        self._last_selected_bbox_center = center
        min_dist = float('inf')
        min_id = 0
        for i in range(NUM_HEADS):
            dist = np.linalg.norm(np.array(center) - np.array(bbox_list[i][:2]))
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id
    
    def select_bbox(self, prob_list, bbox_list, class_list):
        prb_sum = sum(prob_list)
        print(f"Selecting bbox with prob sum: {prb_sum}", flush=True)
        if prb_sum > 0:
            prob_list = [p / prb_sum for p in prob_list]
        else:            
            prob_list = [1.0 / NUM_HEADS] * NUM_HEADS
            
        _id = np.random.choice(len(prob_list), p=prob_list)
        return bbox_list[_id], class_list[_id]
    
    def send_bbox(self, bbox, class_embedded):
        if bbox[0] == 0 and bbox[1] == 0:
            return
        
        bbox_msg = BoundingBox()
        bbox_msg.x = int(bbox[0])
        bbox_msg.y = int(bbox[1])
        bbox_msg.w = int(bbox[2])
        bbox_msg.h = int(bbox[3])
        bbox_msg.class_id = ""
        bbox_msg.confidence = 1.0
        self.bbox_publisher.publish(bbox_msg)
    
    def send_skill(self, skill_vec):
        skill_msg = Float64MultiArray()
        skill_msg.data = skill_vec
        self.skill_vector_publisher.publish(skill_msg)
