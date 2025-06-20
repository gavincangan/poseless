import io
import json
import os
import time

import mujoco
import numpy as np
from datasets import Dataset, Features
from datasets import Image as DsImage
from datasets import Value
from huggingface_hub import HfApi
from PIL import Image
from tqdm.auto import tqdm
from utils import load_hand_model, build_system_prompt

# Remove hard-coded model initialisation – these will be loaded at runtime
model = None  # placeholder, populated in __main__
data = None
renderer = None
joint_names = []
joint_name_to_index = {}
SYSTEM_PROMPT = ""  # will be built dynamically


def get_n_pose_and_upload(
    n,
    dataset_name="hand-poses-dataset",
    push_to_hub=True,
    num_test_sample=1000,
    output_dir="data",
):
    assert (
        num_test_sample < n
    ), "The number of test samples must be lower than the total synthetic subset"
    """Generate n random hand poses and upload to Hugging Face."""

    global joint_names, joint_name_to_index, SYSTEM_PROMPT

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data structures for the dataset
    images_data = []
    joint_positions = []
    filenames = []
    index_mapping = {}

    pose_num = 0
    pbar = tqdm(total=n, desc="Generating poses", unit="pose")
    while len(filenames) < n:
        mujoco.mj_resetData(model, data)

        # Generate random joint positions
        target_positions = []
        for i in range(2, model.njnt):
            joint = model.joint(i)
            target_pos = np.random.uniform(joint.range[0], joint.range[1])
            target_positions.append((joint.qposadr[0], target_pos))

        # Apply control to reach target positions
        for step in range(2000):
            for qpos_addr, target_pos in target_positions:
                current_pos = data.qpos[qpos_addr]
                kp = 10.0  # Proportional gain
                error = target_pos - current_pos
                data.qfrc_applied[qpos_addr] = kp * error

            # Step physics
            mujoco.mj_step(model, data)

            # Check if close enough to targets
            if step % 100 == 0:
                total_error = 0
                for qpos_addr, target_pos in target_positions:
                    total_error += abs(target_pos - data.qpos[qpos_addr])
                if total_error < 0.1 or (step > 500 and data.ncon < 10):
                    break

        # Forward kinematics and render
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="closeup")
        pixels = renderer.render()

        buf = io.BytesIO()
        image = Image.fromarray(pixels)
        image.save(buf, format="PNG")
        buf.seek(0)

        image_filename = f"pose_{pose_num}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)

        # Build angle list in the order of joint_names for _index.json
        angle_list = [round(data.qpos[joint_name_to_index[name]], 4) for name in joint_names]
        index_mapping[image_filename] = angle_list

        filenames.append(image_filename)
        joint_positions.append(data.qpos.copy().tolist())

        buf.close()
        image.close()
        pose_num += 1
        pbar.update(1)  # Update the main progress bar

    pbar.close()  # Close the main progress bar
    print("Processing image data and creating conversations...")

    # Create conversations with tqdm progress
    conversations = []
    for i in tqdm(range(len(filenames)), desc="Creating conversations", unit="conv"):
        # Format each joint angle with the special token format
        joint_description = ""
        for name in joint_names:
            if name in joint_name_to_index:
                joint_idx = joint_name_to_index[name]
                if joint_idx < len(joint_positions[i]):
                    angle_value = round(joint_positions[i][joint_idx], 2)
                    joint_description += f"<{name}>{angle_value}</{name}>"
        conversation = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "content": f"{output_dir}/{filenames[i]}"},
                    {"type": "text", "content": "<Pose>"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "content": f"{joint_description}"},
                ],
            },
        ]
        conversations.append(conversation)

    conversations_json = [json.dumps(conv) for conv in conversations]
    output_path = os.path.join(output_dir, "conversations_dataset.jsonl")
    with open(output_path, "w") as f:
        for conv_json in conversations_json:
            f.write(conv_json + "\n")

    for image_path in filenames:
        images_data.append(Image.open(os.path.join(output_dir, image_path)))

    dataset_dict = {
        "image": images_data,
        "conversations": conversations_json,
    }

    # Create the Hugging Face dataset
    features = Features(
        {
            "image": DsImage(),
            "conversations": Value("string"),
        }
    )

    print("Building dataset object...")
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Save metadata as JSON
    print("Creating metadata...")
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    metadata = {
        "joint_names": model_joint_names,
        "dataset_description": "Random hand poses generated with MuJoCo",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_poses": n,
    }

    with open(os.path.join(output_dir, "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save legacy index/name files
    with open(os.path.join(output_dir, "_index.json"), "w") as f:
        json.dump(index_mapping, f, indent=2)

    with open(os.path.join(output_dir, "_name.json"), "w") as f:
        json.dump(joint_names, f, indent=2)

    if push_to_hub:
        try:
            print(f"Pushing dataset to Hugging Face Hub as {dataset_name}")
            dataset = dataset.train_test_split(test_size=num_test_sample)
            dataset.push_to_hub(
                dataset_name,
            )

            # push the metadata
            api = HfApi()
            print("Uploading metadata...")
            api.upload_file(
                path_or_fileobj=os.path.join(output_dir, "_metadata.json"),
                path_in_repo="_metadata.json",
                repo_id=dataset_name,
                repo_type="dataset",
            )
            print("Upload successful!")
        except Exception as e:
            print(f"Error uploading to Hugging Face Hub: {e}")
            print("Saving dataset locally instead")
            dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
    else:
        print("Saving dataset locally")
        with tqdm(total=100, desc="Saving locally", unit="%") as pbar:
            dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
            pbar.update(100)

    # Clean up resources
    print("Cleaning up resources...")
    for img in images_data:
        try:
            img.close()
        except:
            pass

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate n random hand poses and upload to Hugging Face."
    )
    parser.add_argument("n", type=int, help="Number of poses to generate")
    parser.add_argument(
        "--hand",
        type=str,
        default="shadow_hand",
        help="Name of the hand model (directory under ./models or explicit XML path)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jan-hq/robotic-hand-poses",
        help="Name of the Hugging Face dataset to create",
    )
    parser.add_argument(
        "--no_push",
        action="store_true",
        help="Don't push to Hugging Face, just save locally",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=1000,
        help="Number of test samples to split",
    )

    args = parser.parse_args()

    # Dynamically load the requested hand model and rebuild globals
    model, data, renderer, joint_names, joint_name_to_index = load_hand_model(args.hand)
    SYSTEM_PROMPT = build_system_prompt(joint_names)

    print(f"Starting generation of {args.n} hand poses for hand '{args.hand}' ...")
    start_time = time.time()

    # Create dedicated output directory under data/
    timestamp = time.strftime("%y%m%d%H%M%S")
    output_dir = os.path.join("data", f"{args.hand}_{timestamp}")

    get_n_pose_and_upload(
        args.n,
        args.dataset_name,
        not args.no_push,
        args.num_test_samples,
        output_dir,
    )

    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
