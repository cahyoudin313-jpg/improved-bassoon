# -*- coding: utf-8 -*-
from huggingface_hub.errors import RepositoryNotFoundError
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
import os
from huggingface_hub import create_repo, snapshot_download, HfApi
from pathlib import Path

max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", 1024))
lora_rank = int(os.environ.get("LORA_RANK", 64))
gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.5))
model_name = "Qwen/Qwen2.5-3B-Instruct"
max_steps = int(os.environ.get("MAX_STEPS", 250))
learning_rate = float(os.environ.get("LEARNING_RATE", 5e-6))
per_device_train_batch_size = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 1))
hf_token = os.environ.get("HF_TOKEN", None)
hf_repo = os.environ.get("HF_REPO", None)

if not hf_repo:
    raise ValueError("HF_REPO environment variable must be set")
create_repo(f"{hf_repo}", token=hf_token, exist_ok=True, private=True)
current_dir = Path(__file__).parent

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = gpu_memory_utilization,
    token=os.environ.get("HF_TOKEN", None),
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

dataset = get_gsm8k_questions()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = learning_rate,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 10,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = 4,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 200,
    max_steps = max_steps,
    save_steps = 25,
    save_total_limit = 3,
    save_on_each_node = True,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = f"{current_dir}/{hf_repo}",
    # Hub integration settings
    push_to_hub = True,
    hub_model_id = f"{hf_repo}",
    hub_strategy = "checkpoint",
    hub_private_repo = True,
    hub_token = hf_token,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

def resume_training(trainer, hf_repo):
    """
    Attempts to resume training from the latest checkpoint in the Hugging Face repo.
    Falls back through previous revisions if the latest checkpoint is corrupted.
    Starts fresh if no valid checkpoint is found.
    """
    api = HfApi()
    output_dir = f"{current_dir}/{hf_repo}"

    def try_load_checkpoint(repo_path):
        try:
            checkpoint_dir = os.path.join(repo_path, "last-checkpoint")
            if os.path.exists(checkpoint_dir):
                print(f"Found last-checkpoint directory at {checkpoint_dir}")
                print(f"Checkpoint directory contents: {os.listdir(checkpoint_dir)}")

                # Add this section to get the step number
                state_file = os.path.join(checkpoint_dir, "trainer_state.json")
                if os.path.exists(state_file):
                    import json
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        print(f"Resuming from step {state.get('global_step', 'unknown')}")

                try:
                    trainer.train(resume_from_checkpoint=checkpoint_dir)
                except ValueError as v:
                    print(f"Failed to resume training: {str(v)}")
                    return False
                return True
            else:
                print(f"No last-checkpoint directory found in {repo_path}")
                return False

        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            return False

    try:
        # First try with the last version
        try:
            repo_path = snapshot_download(
                repo_id=hf_repo,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"Cloned repository to {repo_path}")
            print(f"Repository contents: {os.listdir(repo_path)}")

            # Try loading from the latest version
            if try_load_checkpoint(repo_path):
                return

            # If latest version fails, try previous versions
            repo_versions = api.list_repo_commits(hf_repo)
            print(f"Found {len(repo_versions)} versions of repository {hf_repo}")

            for commit in repo_versions[1:]:  # Skip first since we already tried it
                print(f"Checking revision {commit.commit_id}")
                try:
                    repo_path = snapshot_download(
                        repo_id=hf_repo,
                        revision=commit.commit_id,
                        local_dir=output_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"Downloaded revision contents: {os.listdir(repo_path)}")

                    if try_load_checkpoint(repo_path):
                        return

                except Exception as e:
                    print(f"Failed to check revision {commit.commit_id}: {str(e)}")
                    continue

            print("No valid checkpoints found in repository history")

        except RepositoryNotFoundError:
            print(f"Repository {hf_repo} not found")
        except Exception as e:
            print(f"Error accessing repository: {str(e)}")
    except Exception as e:
        print(f"Error in resume process: {str(e)}")

    # If we get here, no checkpoint worked or repo doesn't exist
    print("Starting training from scratch")
    trainer.train()

# Usage
resume_training(trainer, hf_repo)
model.save_lora("grpo_saved_lora")
# Save the final model to the main repo
model.push_to_hub_merged(f"{hf_repo}", tokenizer, save_method="merged_16bit", token=hf_token)
