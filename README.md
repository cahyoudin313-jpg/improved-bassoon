[![Machine](./docs/github-repo-banner.png)](https://machine.dev/)

Machine supercharges your GitHub Workflows with seamless GPU acceleration. Say goodbye to the tedious overhead of managing GPU runners and hello to streamlined efficiency. With Machine, developers and organizations can effortlessly scale their AI and machine learning projects, shifting focus from infrastructure headaches to innovation and speed.


# GRPO Fine-Tuning for LLMs

This repository provides a complete, automated workflow for GPU-accelerated fine-tuning of large language models (LLMs) using Group Relative Policy Optimization (GRPO). It leverages unsloth for efficient fine-tuning, Hugging Face for model hosting, and GitHub Actions powered by Machine.dev for GPU-accelerated training.

We have followed the guides provided by unsloth from their [blog](https://unsloth.ai/blog/r1-reasoning) 
and [Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb)

> **ℹ️ Info:** The code in this repo was taken from the unsloth repository and is used
for the training process. The code is a great resource for understanding the training
process and the techniques used to fine-tune the model.

---

### ✨ **Key Features**

- **⚡ GPU Acceleration:**  Efficiently fine-tune models using GPUs via [Machine](https://machine.dev)
- **💵 Spot Pricing:** Leverage spot-priced GPU instances globally, optimizing performance and cost
- **🚀 GRPO Implementation:** Fine-tune models with Group Relative Policy Optimization techniques
- **💪 Unsloth :** Leverage unsloth's optimizations for faster and more efficient training
- **🔄 Auto-Retry Functionality:** Automatically resume training from checkpoints on spot instance interruptions
- **📤 Hugging Face Hub:** Seamlessly push models and checkpoints to Hugging Face Hub
- **🧮 Math Problem Solving:** Train models to improve performance on mathematical reasoning (GSM8K dataset)
- **🏗️ Format Enforcement:** Enforce structured XML output format for better reasoning traces

---

### 📁 **Repository Structure**

```
├── .github/workflows/
│   ├── training.yaml                    # Basic training workflow
│   └── training-with-retry.yaml         # Training workflow with retry/checkpointing
├── .github/actions/check-runner-interruption/
│   └── action.yaml                      # Action to detect spot instance interruptions
├── qwen2_5_(3b)_grpo.py                 # Main GRPO fine-tuning script
├── qwen2_5_(3b)_grpo_checkpointing.py   # Extended script with checkpointing support
└── requirements.txt                     # Python dependencies
```

### ▶️ **Getting Started**

#### 1. **Use This Repository as a Template**
Click the **Use this template** button at the top of this page to create your own copy.

#### 2. **Set Up GPU Runners**
Ensure your repository uses Machine GPU-powered runners. No additional configuration is required if you're already using Machine.dev.

#### 3. **Configure Hugging Face Access**

1. Create a Hugging Face access token with write permissions
2. Add this token as a repository secret named `HF_TOKEN` in your GitHub repository settings

#### 4. **Run the Workflow**

- Trigger the workflow manually in GitHub Actions (workflow_dispatch)
- The workflow allows you to customize training parameters such as:

  - Maximum sequence length
  - LoRA rank
  - Maximum training steps
  - GPU memory utilization
  - Learning rate
  - Batch size
  - Hugging Face repository to push to

You can choose between two workflows:

- `training.yaml`: Basic training without checkpointing
- `training-with-retry.yaml`: Training with automatic checkpointing and retry on spot instance interruptions

##### Basic Training Workflow Parameters

The `training.yaml` workflow accepts the following inputs:

```yaml
inputs:
  max_seq_length:
    type: string
    required: false
    description: 'The maximum sequence length'
    default: '1024'
  lora_rank:
    type: string
    required: false
    description: 'The lora rank'
    default: '64'
  max_steps:
    type: string
    required: false
    description: 'The maximum number of steps'
    default: '250'
  gpu_memory_utilization:
    type: string
    required: false
    description: 'The GPU memory utilization'
    default: '0.60'
  learning_rate:
    type: string
    required: false
    description: 'The learning rate'
    default: '5e-6'
  per_device_train_batch_size:
    type: string
    required: false
    description: 'The per device training batch size'
    default: '2'
  hf_repo:
    type: string
    required: false
    description: 'The Hugging Face repository to upload the model to'
```

##### Training with Retry Workflow Parameters

The `training-with-retry.yaml` workflow includes all the parameters from the basic training workflow plus:

```yaml
inputs:
  attempt:
    type: string
    description: 'The attempt number'
    default: '1'
  max_attempts:
    type: number
    description: 'The maximum number of attempts'
    default: 5
  # (All parameters from the basic training workflow are also included)
```

##### How the Retry Mechanism Works

The retry mechanism is designed to handle the potential interruption of training jobs running on spot instances. Here's how it works:

1. The workflow starts a training job with a specified attempt number (default: 1)
2. If the job completes successfully, the workflow ends
3. If the job fails due to a spot instance interruption:
   - The `check-runner-interruption` action detects that the failure was due to a spot instance preemption
   - The workflow calculates the next attempt number
   - If the next attempt is within the maximum attempts limit, it triggers a new workflow run with an incremented attempt number
   - All original parameters are preserved for the new attempt
4. The training script (`qwen2_5_(3b)_grpo_checkpointing.py`) automatically saves checkpoints to a Hugging Face repository during training
5. When a new attempt starts, it tries to download the latest checkpoint and resume training from that point

This mechanism ensures that even if a spot instance is reclaimed, your training progress isn't lost, and the job can continue from the last checkpoint on a new instance.

#### 5. **Monitor and Review Results**

- Training progress is logged during the workflow execution
- The trained model is automatically pushed to your specified Hugging Face repository
- Checkpoints are saved to a separate repository (your-repo-name-checkpoints) when using the retry workflow

---

### 🔑 **Prerequisites**


- GitHub account
- Access to [Machine](https://machine.dev) GPU-powered runners
- [Hugging Face](https://huggingface.co) account for model hosting

_No local installation necessary—all processes run directly within GitHub Actions._

---

### 📄 **License**

This repository is available under the [MIT License](LICENSE).

---

### 📌 **Notes**

- This fine-tuning template specifically targets the Qwen 2.5 3B model and trains it using GRPO on the GSM8K dataset, focusing on improving mathematical reasoning with structured output format. The repository can be adapted for other models, datasets, and tasks with minimal modifications.

- This repository is currently open for use as a template. While public forks are encouraged, we are not accepting Pull Requests at this time.

_For questions or concerns, please open an issue._

