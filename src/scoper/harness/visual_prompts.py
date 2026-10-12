from scoper.harness import recovery_prompts
from scoper.harness.prompt_contracts import VERIFIER_V11_SYSTEM_BYTES
FRAME_GUIDANCE = b'Action verification uses two separate ordered images: BEFORE (pre_rgb), the view before execution, and AFTER (post_rgb), the immediate view after execution. No extra camera movement is performed for verification. Unclear or occluded effects require ambiguous, not failure. An unchanged image alone does not prove failure. For placement, require visible evidence of the object in or on the requested target; being near the target alone is insufficient.\n'
POLICY_SYSTEM_BYTES = recovery_prompts.POLICY_SYSTEM_BYTES + FRAME_GUIDANCE
VERIFIER_SYSTEM_BYTES = VERIFIER_V11_SYSTEM_BYTES + FRAME_GUIDANCE + recovery_prompts.GRIPPER_MARKER_GUIDANCE_BYTES
MEMORY_SYSTEM_BYTES = b'\n'.join((line for line in recovery_prompts.MEMORY_SYSTEM_BYTES.split(b'\n') if b'The three image keys have fixed meanings:' not in line)) + FRAME_GUIDANCE
D6_POLICY_SYSTEM_BYTES = POLICY_SYSTEM_BYTES + recovery_prompts.D6_POLICY_SYSTEM_BYTES[len(recovery_prompts.POLICY_SYSTEM_BYTES):]
