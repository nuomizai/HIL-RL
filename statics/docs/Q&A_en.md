This Markdown file temporarily collects answers to a few common questions (translated and polished by an LLM, re-checked by human).


---

## Q1: About the Action Space

**A1:** In this project, “actions” appear in three places. They have the same semantics, but are **handled differently**:

1. **Policy network action output**: `policy.select_action` in `actor.py`
2. **Human intervention action output**: `HumanIntervention.action` in `rl_envs/wrappers.py`
3. **Action received by the environment**: `step` in `rl_envs/base_env.py`

### (1) / (2) Policy and human intervention actions (normalized scale)

* **Dimension**

  * 6 (without gripper) or 7 (with gripper)
* **Meaning**

  * Relative end-effector motion **scale** in the gripper frame: `x, y, z, r, p, y` (+ optional gripper)
* **Range**

  * `ee_pose`: `(-1, 1)`
  * `gripper`: `0/1`

Human intervention actions share the **exact same semantics** as policy actions. During intervention, the system first computes the physical relative displacement of the isomorphic arm, then divides by the maximum range of each dimension to obtain the normalized values. The default ranges are typically:

* Translation (`x, y, z`): up to `0.02 m`
* Rotation (`r, p, y`): up to `0.06 rad`

These ranges can be modified in the corresponding YAML files under `cfg/robot_type/`.

### (3) Actions received by the environment (normalized scale)

* **Dimension**: 7
* **Meaning**: Same as (1) and (2)

In `step`, the `action` is interpreted as the **physical** relative end-effector motion in the gripper frame (units are typically meters/radians). Therefore, in `step`, the normalized action from (1) or (2) must be scaled back to physical values before being sent to the robot.