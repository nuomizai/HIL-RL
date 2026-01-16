This Markdown file temporarily collects answers to a few common questions (translated and polished by an LLM, re-checked by human).


---

## Q1: Action Space

**A1:** In this project, “actions” appear in three places. They represent the same intent, but are **encoded differently**:

1. **Policy output action**: `policy.select_action` in `actor.py`
2. **Human intervention action**: `HumanIntervention.action` in `rl_envs/wrappers.py`
3. **Environment input action**: `step` in `rl_envs/base_env.py`

#### (1) / (2) Policy & human intervention actions (normalized scale)

* **Dimension**:

  * 6 (without gripper) or 7 (with gripper)
* **Meaning**: Relative motion **scale** of the end-effector in the gripper (EE) frame: `x, y, z, r, p, y` (+ optional gripper)
* **Range**:

  * `ee_pose`: `(-1, 1)`
  * `gripper`: `0/1`

Human intervention actions have exactly the same semantics as policy actions. During intervention, the physical relative offset of the isomorphic arm is first computed, then divided by the maximum range of each dimension to obtain the normalized value. Typical default ranges are:

* Translation (`x, y, z`): max `0.02 m`
* Rotation (`r, p, y`): max `0.06 rad`

These ranges can be modified in the corresponding YAML files under `cfg/robot_type/`.

#### (3) Environment input action (physical values)

* **Dimension**: 7
* **Meaning**: Physical relative motion of the end-effector in the gripper (EE) frame (units are typically meters/radians)

Therefore, actions in (1) or (2) must be scaled by the corresponding ranges before being passed to the environment:

`action_physical = action_scale * action_range`
