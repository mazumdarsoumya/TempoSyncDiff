<div align="center">

# TempoSyncDiff

### Distilled Temporally-Consistent Diffusion for Low-Latency Audio-Driven Talking Head Generation

</div>

Welcome.

This repository contains the official implementation of **TempoSyncDiff**, a framework for **audio-driven talking-head generation** designed to achieve **low-latency inference** while maintaining **temporal coherence and identity consistency**.

TempoSyncDiff explores whether **diffusion models can be distilled into efficient few-step generators** without substantially degrading visual fidelity or motion stability.


---

# 🔗 Project Links

* **Paper (arXiv)**  
  [arXiv:2603.06057v1](https://doi.org/10.48550/arXiv.2603.06057)

* **GitHub Repository**  
  [https://github.com/mazumdarsoumya/TempoSyncDiff](https://github.com/mazumdarsoumya/TempoSyncDiff.git)

* **Project Page**  
  [https://mazumdarsoumya.github.io/TempoSyncDiff](https://mazumdarsoumya.github.io/TempoSyncDiff)

---

# 📄 Paper

**Title**

*TempoSyncDiff: Distilled Temporally-Consistent Diffusion for Low-Latency Audio-Driven Talking Head Generation*

The manuscript is currently **submitted for peer review**.

Until formal publication, the **arXiv version serves as the primary public reference**.

---

# 🧠 Abstract

Audio-driven talking-head synthesis aims to generate photorealistic facial animations aligned with speech signals. While diffusion-based generative models have demonstrated strong synthesis quality, their computational cost often limits real-time or low-latency deployment.

TempoSyncDiff investigates a **teacher–student distillation framework for latent diffusion models** that reduces the number of denoising steps required during inference. The approach integrates **identity anchoring**, **temporal regularization**, and **viseme-conditioned control** to maintain identity consistency and reduce frame-to-frame instability.

The resulting distilled student model enables **few-step latent diffusion inference**, offering improved efficiency while preserving visual quality and temporal coherence.

---

# 🧠 Overview

Diffusion models are capable of impressive visual synthesis, although they are occasionally fond of taking their time. TempoSyncDiff explores whether **teacher–student distillation** can preserve much of the denoising capability of a stronger diffusion model while enabling **fast few-step inference**.

The framework incorporates:

- **Identity anchoring** for identity preservation
- **Temporal regularization** for improved frame-to-frame stability
- **Viseme-based conditioning** for speech-aligned lip motion
- **Latent diffusion modeling** for computational efficiency
- **Distilled few-step sampling** for low-latency generation

The method aims to balance three key objectives:

1. **Visual realism**
2. **Temporal stability**
3. **Efficient inference**

Achieving all three simultaneously remains a respectable negotiation with both GPUs and the laws of optimization.

---

# ⚙️ Method Pipeline

TempoSyncDiff follows a **latent diffusion training and distillation pipeline**:

1. **Frame Compression**

   Video frames are encoded into a compact latent space using a lightweight **Variational Autoencoder (VAE)**.

2. **Teacher Diffusion Training**

   A **latent diffusion teacher model** learns to predict noise across the diffusion process.

3. **Student Distillation**

   A smaller **student denoiser** is trained to approximate the teacher using **fewer denoising steps**.

4. **Few-Step Inference**

   The distilled student model performs **efficient video frame generation** conditioned on identity and speech-derived tokens.

Additional training components include:

- **Identity anchor module**
- **Temporal smoothness losses**
- **Viseme token conditioning**

---

# 📦 What This Repository Includes

This repository provides:

- Complete **reference project structure**
- Source code for
  - VAE pretraining
  - teacher diffusion training
  - student distillation
  - inference
  - evaluation
- configuration files for LRS3-style experiments
- scripts for dataset manifest generation
- placeholder directories for datasets and checkpoints

This repository **does not include**:

- trained model checkpoints
- restricted datasets
- generated experiment archives

---

# 🗂 Repository Structure

```

TempoSyncDiff/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── checkpoints/
│   └── README.md
│
├── configs/
│   ├── pretrain/
│   │   └── tinyvae_lrs3.yaml
│   ├── train/
│   │   ├── teacher_lrs3.yaml
│   │   └── student_distill_lrs3.yaml
│   ├── infer/
│   │   └── student_infer.yaml
│   └── eval/
│       ├── denoise_eval.yaml
│       └── latency_cpu.yaml
│
├── data/
│   ├── README.md
│   ├── lrs3/
│   ├── hdtf/
│   ├── manifests/
│   ├── visemes/
│   └── examples/
│
├── docs/
│   └── RELEASE_NOTES.md
│
├── outputs/
│   ├── logs/
│   ├── samples/
│   ├── metrics/
│   ├── plots/
│   └── tables/
│
├── scripts/
│   ├── build_lrs3_manifest.py
│   ├── pretrain_vae.py
│   ├── train_teacher.py
│   ├── distill_student.py
│   ├── infer_student.py
│   └── evaluate.py
│
└── src/
└── temposyncdiff/
├── data/
├── losses/
├── models/
└── utils/

```

---

# 📊 Datasets

Experiments primarily involve:

- **LRS3-TED**
- **HDTF**

Please obtain these datasets from their **official distribution sources** and comply with their licenses and usage terms.

---

# 📁 Expected Data Layout

### LRS3 Dataset Structure

```

data/lrs3/
└── <talk_id>/
├── 00001.mp4
├── 00001.txt
├── 00002.mp4
├── 00002.txt

```

Optional viseme token files:

```

data/visemes/lrs3/
└── <talk_id>/
├── 00001.npy
├── 00002.npy

````

If viseme tokens are unavailable, the pipeline **automatically falls back to zero-token conditioning**, allowing training and inference to remain executable.

---

# 💻 Environment Setup

Python **3.10+** is recommended.

### macOS / Linux

```bash
git clone https://github.com/mazumdarsoumya/TempoSyncDiff.git
cd TempoSyncDiff

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e .
````

### Windows

```powershell
git clone https://github.com/mazumdarsoumya/TempoSyncDiff.git
cd TempoSyncDiff

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

# 🧪 Training Workflow

## 1 — Build Dataset Manifests

```bash
python scripts/build_lrs3_manifest.py \
  --root data/lrs3 \
  --out_dir data/manifests \
  --viseme_root data/visemes/lrs3 \
  --val_ratio 0.01 \
  --test_ratio 0.01 \
  --seed 123
```

Generated files:

```
data/manifests/lrs3_train.jsonl
data/manifests/lrs3_val.jsonl
data/manifests/lrs3_test.jsonl
```

---

## 2 — Pretrain the VAE (Optional)

```bash
python scripts/pretrain_vae.py \
  --config configs/pretrain/tinyvae_lrs3.yaml
```

Outputs:

```
checkpoints/vae_pretrained.pt
```

---

## 3 — Train the Teacher Diffusion Model

```bash
python scripts/train_teacher.py \
  --config configs/train/teacher_lrs3.yaml
```

Outputs:

```
checkpoints/teacher_best.pt
checkpoints/teacher_last.pt
```

---

## 4 — Distill the Student Model

```bash
python scripts/distill_student.py \
  --config configs/train/student_distill_lrs3.yaml
```

Outputs:

```
checkpoints/student_best.pt
checkpoints/student_last.pt
```

---

# 🎬 Inference

Place a reference image:

```
data/examples/ref.jpg
```

Run:

```bash
python scripts/infer_student.py \
  --config configs/infer/student_infer.yaml
```

Outputs:

```
outputs/samples/<run_name>/frames/
outputs/samples/<run_name>/sample.mp4
```

---

# 📈 Evaluation

Evaluation compares denoised outputs against **VAE reconstructions**.

```bash
python scripts/evaluate.py \
  --config configs/eval/denoise_eval.yaml
```

Output:

```
outputs/metrics/eval_report.json
```

---

# ⚡ Quick Inference (Without Training)

1. Install dependencies
2. Place checkpoints in `checkpoints/`
3. Add reference image:

```
data/examples/ref.jpg
```

Run:

```bash
python scripts/infer_student.py \
  --config configs/infer/student_infer.yaml
```

---

# 📜 Citation

If this work contributes to your research, please cite:

```bibtex
@article{mazumdar2026temposyncdiff,
  title={TempoSyncDiff: Distilled Temporally-Consistent Diffusion for Low-Latency Audio-Driven Talking Head Generation},
  author={Mazumdar, Soumya and Rakesh, Vineet Kumar},
  journal={arXiv preprint arXiv:2603.06057},
  year={2026},
  doi={10.48550/arXiv.2603.06057}
}
```

---

# 🤝 Acknowledgements

This research benefited from institutional support and infrastructure provided by:

* **Variable Energy Cyclotron Centre (VECC)**
* **Department of Atomic Energy (DAE), Government of India**

The authors also acknowledge the broader research community whose work continues to advance generative modeling.

---

# ⚠️ Responsible Use

Talking-head generation systems should be used responsibly.

Users are encouraged to:

* obtain appropriate consent
* respect dataset licenses
* clearly disclose synthetic media when applicable

---

# 📬 Contact

**Soumya Mazumdar**
[reachme@soumyamazumdar.com](mailto:reachme@soumyamazumdar.com)

**Vineet Kumar Rakesh**
[vineet@vecc.gov.in](mailto:vineet@vecc.gov.in)
