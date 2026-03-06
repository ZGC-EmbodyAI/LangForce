<div align="center">

<h1>
    <span
        style="
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            display: inline-block;
        "
    >LangForce</span>
    : Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
</h1>

<a href="https://github.com/ZGC-EmbodyAI/LangForce">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-ZGC--EmbodyAI%2FLangForce-blue?logo=github">
</a>
<a href="https://www.alphaxiv.org/abs/2601.15197">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.15197-b31b1b.svg">
</a>
<a href="https://github.com/ZGC-EmbodyAI/TwinBrainVLA/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</a>

**Shijie Lian**<sup>1,2,\*</sup> **Bin Yu**<sup>2,4,\*</sup> **Xiaopeng Lin**<sup>2,5,\*</sup> **Laurence T. Yang**<sup>6,1,†</sup> **Zhaolong Shen**<sup>2,7</sup><br>
**Changti Wu**<sup>2,8</sup> **Yuzhuo Miao**<sup>2,4</sup> **Cong Huang**<sup>2,3</sup> **Kai Chen**<sup>2,3,9,†</sup>

<sup>1</sup>HUST, <sup>2</sup>ZGCA, <sup>3</sup>ZGCI, <sup>4</sup>HIT, <sup>5</sup>HKUST(GZ), <sup>6</sup>ZZU, <sup>7</sup>BUAA, <sup>8</sup>ECNU, <sup>9</sup>DeepCybo

<sup>*</sup>Equal contribution, <sup>†</sup>Corresponding author

<img src="./assets/ZGCA-logo.png" alt="ZGCA" style="vertical-align: middle; height: 16px; margin-right: 4px; position: relative; top: -2px;" />[Zhongguancun Academy](https://www.bjzgca.edu.cn/) & <img src="./assets/ZGCI-logo.png" alt="ZGCI" style="vertical-align: middle; height: 16px; margin-right: 4px; position: relative; top: -2px;" />[Zhongguancun Institute of Artificial Intelligence](https://www.zgci.ac.cn/)

</div>

---

## 📢 News

- [Feb 10, 2026] :zap: LangForce has been integrated into [starVLA](https://github.com/starVLA/starVLA). You can now directly train LangForce through starVLA and perform end-to-end training and evaluation on benchmarks such as LIBERO, SimplerEnv, and RoboCasa.


## 📖 Abstract

Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term `Information Collapse`. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose **LangForce:**, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable **Latent Action Queries**, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $\pi(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, LangForce significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an **11.3\%** improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.

## 🏗️ Architecture

**LangForce** is a novel framework designed to solve the **Vision Shortcut** problem in Vision-Language-Action (VLA) models. 

<div align="center">
  <img src="./assets/arch.png" alt="LangForce Framework" width="100%">
</div

In current VLA training, goal-driven datasets often make language instructions highly predictable from visual observations alone. This leads to **Information Collapse**, where the model ignores language and degenerates into a vision-only policy, failing miserably in out-of-distribution (OOD) scenarios.

**LangForce** addresses this by:
1. **Bayesian Decomposition**: Explicitly modeling a vision-only prior $p(a|v)$ and a language-conditioned posterior $\pi(a|v, \ell)$.
2. **LLR Optimization**: Maximizing the Log-Likelihood Ratio (LLR) to penalize actions that rely solely on visual cues and reward actions that are truly grounded in language instructions.

## ✨ Key Features

- **Dual-Branch Architecture**: Uses learnable **Latent Action Queries** to decouple vision-only and language-conditioned action distributions.
- **Zero Extra Data**: Achieves significant performance gains (e.g., **+11.3%** on SimplerEnv) using the exact same datasets as baselines.
- **Preserves VLM Intelligence**: Effectively regularizes the model to prevent the "catastrophic forgetting" of general multimodal reasoning capabilities common in standard VLA fine-tuning.

## 📊 Performance

| Method | SimplerEnv (Avg) | RoboCasa (Avg) | LIBERO (Avg) |
| :--- | :---: | :---: | :---: |
| QwenGR00T (Baseline) | 55.2% | 47.8% | 96.5% | 
| **LangForce (Ours)** | **66.5% (+11.3%)** | **52.6% (+4.8%)** | **98.4% (+1.9%)** |

## 🚀 Training

Our training pipeline is built upon the **StarVLA** framework. To get started, please follow the instructions below to set up the base environment.

<details close>
<summary><b>🛠 starVLA Environment Setup
</b></summary>

```bash
# Clone the repo
git clone https://github.com/starVLA/starVLA

# Create conda environment
conda create -n starVLA python=3.10 -y
conda activate starVLA

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install starVLA
pip install -e .
```

⚠️ **Common Issues**
flash-attn can be tricky to install because it must match your system’s CUDA toolkit (nvcc) and PyTorch versions. The `--no-build-isolation` flag resolves most issues, but on newer systems you may need to manually choose a compatible flash-attn version. Ensure your CUDA driver/toolkit and torch versions are aligned. Check your environment:

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

If issues persist, pick a flash-attn release that matches your versions (CUDA and torch) or ask chatGPT with searching function for help with the outputs above.

We have verified that `flash-attn==2.7.4.post1` works well with nvcc versions `12.0` and `12.4`.

</details>

**Train**

1. **Vocabulary Expansion**: LangForce utilizes Qwen3-VL and extends the vocabulary with specialized tokens that serve as Latent Action Queries. Run the provided example script `add_token.py` to update the tokenizer with these additional tokens.

2. **Training Script**: You can learn how to train LangForce using starVLA from [here]{https://github.com/starVLA/starVLA?tab=readme-ov-file#-quick-start}. Below, we provide a training script for LangForce on 8 × H100 GPUs:

```bash
conda activate starvla
cd /xxx/worlkplace/starVLA-v2.0

export NCCL_SOCKET_IFNAME=eth0        
export NCCL_IB_DISABLE=1       
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

framework_name=LangForceV5
base_vlm=/xxx/starVLA-v2.0/playground/Pretrained_models/Qwen3-VL-4B-with-Action-Query
run_id=GR00T_Simpler_LangForce
freeze_module_list=''
config_yaml=./examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
oxe_data_root=/xxx/starVLA/playground/Datasets/OXE_LEROBOT_DATASET/
data_mix=bridge
run_root_dir=./results/LangForce/SimplerEnv

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.template ${vlm_template} \
  --framework.detach_prior_cond ${detach_prior_cond} \
  --framework.qwenvl.num_latent_action_query ${num_latent_action_query} \
  --framework.action_model.diffusion_model_cfg.num_layers ${dit_num_layers} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity xxx
```

> LangForce is currently under active development. Feel free to check back frequently for updates and new features!

## 🙏 Acknowledgements

We would like to thank the [starVLA](https://github.com/starVLA/starVLA) project for its inspiring work and open-source contributions. At the same time, we also express our gratitude to the following projects:

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot](https://github.com/huggingface/lerobot/)
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
- [Franka Teleop](https://github.com/Shenzhaolong1330/lerobot_franka_teleop)

## Citation
If you find this project or the dataset helpful, please cite:
```bibtex
@misc{LangForce_2026_arXiv,
      title={LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries}, 
      author={Shijie Lian and Bin Yu and Xiaopeng Lin and Laurence T. Yang and Zhaolong Shen and Changti Wu and Yuzhuo Miao and Cong Huang and Kai Chen},
      year={2026},
      eprint={2601.15197},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.15197}, 
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ZGC-EmbodyAI/LangForce&type=date&legend=top-left)](https://www.star-history.com/#ZGC-EmbodyAI/LangForce&type=date&legend=top-left)
