import os
from typing import List, Tuple

import mujoco


def locate_hand_xml(hand_name: str) -> str:
    """Resolve the path to a MJCF (.xml) file for the given hand.

    The function supports the following input conventions:
    1. If *hand_name* already points to an existing *.xml* file, it is returned as-is.
    2. If *hand_name* is a directory, the first *.xml* file found inside it is returned.
    3. Otherwise the function searches inside the repository *models/* directory for a
       sub-directory matching *hand_name* and returns the first *.xml* file within it.

    Raises:
        FileNotFoundError: If no suitable *.xml* file can be located.
    """
    # Case 1 – user passed an explicit XML path
    if os.path.isfile(hand_name) and hand_name.endswith(".xml"):
        return hand_name

    # Case 2 – user passed a directory path directly
    if os.path.isdir(hand_name):
        for fn in os.listdir(hand_name):
            if fn.endswith(".xml"):
                return os.path.join(hand_name, fn)

    # Case 3 – look inside ./models/<hand_name>
    repo_root = os.path.dirname(os.path.abspath(__file__))
    hand_dir = os.path.join(repo_root, "models", hand_name)
    if os.path.isdir(hand_dir):
        for fn in os.listdir(hand_dir):
            if fn.endswith(".xml"):
                return os.path.join(hand_dir, fn)

    raise FileNotFoundError(
        f"Could not find a .xml file for hand '{hand_name}'. "
        "Pass an explicit path or ensure the hand exists under ./models/"
    )


def load_hand_model(
    hand_name: str,
    width: int = 1280,
    height: int = 1280,
) -> Tuple[mujoco.MjModel, mujoco.MjData, mujoco.Renderer, List[str], dict]:
    """Load a MuJoCo hand model and return useful artefacts.

    Returns a tuple consisting of
        (model, data, renderer, joint_names, joint_name_to_index)
    """
    xml_path = locate_hand_xml(hand_name)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Build joint list while skipping virtual joints
    joint_names: List[str] = []
    joint_name_to_index: dict = {}

    for i in range(model.njnt):
        j = model.joint(i)
        name = j.name
        if name is None or name.endswith("_virt"):
            # Skip virtual or unnamed joints
            continue
        joint_names.append(name)
        # Map directly to the qpos index inside MuJoCo's state vector
        joint_name_to_index[name] = j.qposadr[0]

    return model, data, renderer, joint_names, joint_name_to_index


def build_system_prompt(joint_names: List[str]) -> str:
    """Create a dynamic system prompt listing all joint tags."""
    fmt = "".join([f"<{name}>angle</{name}>" for name in joint_names])
    return (
        "You are a specialized Vision Language Model designed to accurately estimate joint angles from hand pose images. "
        "Your task is to analyze images of a human or robotic hand and output precise angle measurements for each joint. "
        "Output joint angles in radians.\nOutput Format:\n" + fmt
    ) 