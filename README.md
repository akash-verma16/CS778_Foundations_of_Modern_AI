# CS778-Foundations-Of-Modern-AI
THis Porject focuses on implementing key RL-Algos like Policy Gradient TRPO and PPO and study them under 5 gym enviroments along with employing a Hybrid Approach to Enhance Human Alignment of LLMs using a combination of Dueling Bandits as well as DPO.

---

# 🚀 Online & MixP DPO Implementation using Pythia-410M

### _Optimized for 8GB VRAM setups – A lightweight adaptation of **Samplers-in-Online-DPO**_

---

## 📌 Overview

This repository presents a minimal, GPU-friendly implementation of **Online Direct Preference Optimization (DPO)** and **MixP-DPO**, adapted to run on **Pythia-410M (410M parameters)** with **8GB VRAM**.

It is **based on the official implementation and paper**:

> **"The Crucial Role of Samplers in Online Direct Preference Optimization"**  
> *Shi et al., ICLR 2025*

---

## 🧠 Key Highlights

- 🔍 Adapted original framework to **Pythia-410M** instead of 7B models
- 📈 Implemented **Online DPO (2 iterations)** and **MixP DPO (1 iteration)**
- 🤖 Used **DeBERTa-v3 reward model** due to GPU constraints
- 📊 All training runs logged on **Weights & Biases**
- 🛠 Modular runnable scripts for full pipeline: **Generation → Annotation → Training**

---

## 📂 Project Structure
```
alignment/
├── data/                          # Generated & annotated data
├── scripts/
│   └── safe_rlhf/
│       ├── gen_online_1b.sh       # Online generation
│       ├── gen_mixp1.sh           # MixP generation
│       ├── dpo_online_1b.sh       # Online DPO training
│       ├── dpo_mixp.sh            # MixP DPO training
│       └── annotate.sh            # Reward annotation
├── generation/
│   └── safe_rlhf/
│       ├── get_hf2.py
│       └── mixp.py                # MixP dataset merger
├── dpo_iteration/
│   └── run_dpo.py                 # Core DPO logic
├── configs/                       # Accelerate & training configs
└── README.md
```

---

## ⚙️ Conda Environments Used

| Stage      | Conda Env    | Purpose              |
|------------|--------------|----------------------|
| Generation | `vllm`       | Fast inference       |
| Annotation | `rewardflow` | Reward model scoring |
| Training   | `rlhflow`    | DPO fine-tuning      |

---

## 🚀 Usage Pipeline

### 🔹 1️⃣ Online DPO — Example for Iteration 2
```bash
# Generation (vllm env)
bash scripts/safe_rlhf/gen_online_1b.sh 2 3 online

# Annotation (rewardflow env)
bash scripts/safe_rlhf/annotate.sh 2 online 3

# Training (rlhflow env)
wandb login
bash scripts/safe_rlhf/dpo_online_1b.sh 2
```

### 🔹 2️⃣ MixP DPO — Example for Iteration 1
```bash
# Generation (vllm env)
bash scripts/safe_rlhf/gen_mixp1.sh 1 4

# If merge needed
python generation/mixp.py \
  --policy ./data/gen_data0_policy.json \
  --ref ./data/gen_data0_ref.json \
  --output ./data/gen_data_iter1_mixp.json

# Annotation (rewardflow env)
bash scripts/safe_rlhf/annotate.sh 1 mixp 4

# Training (rlhflow env)
wandb login
bash scripts/safe_rlhf/dpo_mixp.sh 1
```

---

## 📊 Results & Observations

| Method     | Model Size | Iterations | Accuracy Trend            |
|------------|------------|------------|---------------------------|
| Online DPO | 410M       | 2          |  Improved accuracy, unstable       |
| MixP DPO   | 410M       | 1          | Good Mix between accuracy and Offline Stability  |

📌 **Detailed charts available in W&B runs:**
```
online_410m_iter1, online_410m_iter2, mixp_410m_iter1
```

---

## 📎 References

### 🔸 Paper Citation
```bibtex
@inproceedings{
  shi2024crucialroleosamplerdpo,
  title={The Crucial Role of Samplers in Online Direct Preference Optimization},
  author={Ruizhe Shi and Runlong Zhou and Simon S. Du},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=F63ztufcKw}
}
```

### 🔸 Original Repository Citation
```bibtex
@software{shi2024samplersgithub,
  author = {Ruizhe Shi and collaborators},
  title = {Samplers-in-Online-DPO},
  year = {2024},
  url = {https://github.com/szze/Samplers-in-Online-DPO},
  note = {Official DPO implementation}
}
```

---

## 🔍 Future Work

- [ ] Extend MixP to multiple iterations
- [ ] Experiment with LoRA to enable larger models
- [ ] Try high-capacity reward models via Triton or CPU offloading

---

## Contributors
- **Aarsh Kaushik** ([@Aarsh59](https://github.com/Aarsh59))
- **Akash Verma**
- **Tanmya Siddharth** 


Indian Institute of Technology Kanpur  
📬 Open to collaboration and discussions!

---

## 🧾 License

MIT License – see [LICENSE](LICENSE) file.

---

## ⭐ Acknowledgments

If this repo helps you, consider giving the [original authors' repository](https://github.com/szze/Samplers-in-Online-DPO) a ⭐ on GitHub and citing their work.
