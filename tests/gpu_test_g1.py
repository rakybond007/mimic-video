"""G1: action_mask가 forward()에서 동작하는지 테스트"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from mimic_video.mimic_video import MimicVideo
from mimic_video.cosmos_predict import CosmosPredictWrapper

wrapper = CosmosPredictWrapper(random_weights=True, tiny=True)
model = MimicVideo(
    dim=64, video_predict_wrapper=wrapper,
    action_chunk_len=16, dim_action=32, dim_joint_state=64,
    num_video_viewpoints=2, depth=2, heads=2, dim_head=32,
).cuda()

B = 2
video = torch.rand(B, 2, 1, 3, 64, 64).cuda()  # tiny VAE expects <=64x64
actions = torch.rand(B, 16, 32).cuda()
joint_state = torch.rand(B, 64).cuda()
mask = torch.zeros(B, 32, dtype=torch.bool).cuda()
mask[:, :7] = True

# action_mask 없이
loss_no_mask = model(actions=actions, joint_state=joint_state, video=video, prompts=["test"] * B)

# action_mask 있으면
loss_with_mask = model(actions=actions, joint_state=joint_state, action_mask=mask, video=video, prompts=["test"] * B)

print(f"Loss without mask: {loss_no_mask.item():.4f}")
print(f"Loss with mask:    {loss_with_mask.item():.4f}")
assert loss_no_mask.shape == (), f"Expected scalar, got {loss_no_mask.shape}"
assert loss_with_mask.shape == (), f"Expected scalar, got {loss_with_mask.shape}"
print("G1 PASSED")
