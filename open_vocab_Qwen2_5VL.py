import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from utils import load_hand_model
import argparse

# we dont have to resize embedding and lm_head cause the number of padding is 271.
device = torch.device("cpu")

# Parse joint names dynamically based on requested hand
parser = argparse.ArgumentParser(
    description="Resize Qwen2.5-VL tokenizer for a specific hand pose task."
)
parser.add_argument(
    "--hand",
    type=str,
    default="shadow_hand",
    help="Name of the hand model (directory under ./models or explicit XML path)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="../Qwen2.5-VL-3B-Instruct-Resized/",
    help="Where to save the resized model/tokenizer",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="jan-hq/Qwen2.5-VL-3B-Instruct-Resized",
    help="Destination HF repo id",
)
args = parser.parse_args()

# Load joint names from hand model
model_tmp, _, _, joint_names, _ = load_hand_model(args.hand)

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
old_vocab_size = len(tokenizer)
print("Original vocab size:", old_vocab_size)

# Task and joint tokens
task_token = ["<Pose>"]
add_tokens = task_token
for name in joint_names:
    add_tokens.append(f"<{name}>")
    add_tokens.append(f"</{name}>")

print("Number of new tokens:", len(add_tokens))

# Extend vocabulary
num_added = tokenizer.add_tokens(add_tokens)
print(f"Added {num_added} tokens to tokenizer")

# Resize model embeddings only if new tokens were actually added
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

# Save locally
output_dir = args.output_dir
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# Push to HF hub
repo_id = args.repo_id
api = HfApi()
create_repo(repo_id, exist_ok=True)
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
)
print("Model and tokenizer updated and pushed to Hugging Face Hub.")
