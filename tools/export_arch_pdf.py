import os, argparse, textwrap, datetime
from pathlib import Path

import torch
import yaml

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Optional: nicer layer summaries (only if model can be built)
try:
    from torchinfo import summary as torchinfo_summary
    _HAS_TORCHINFO = True
except Exception:
    _HAS_TORCHINFO = False


def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def state_dict_stats(sd: dict):
    if not isinstance(sd, dict):
        return (0, 0)
    tensors = 0
    params = 0
    for _, v in sd.items():
        if torch.is_tensor(v):
            tensors += 1
            params += v.numel()
    return tensors, params


def flatten_cfg(cfg: dict, prefix=""):
    items = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            items.extend(flatten_cfg(v, kp))
    elif isinstance(cfg, (list, tuple)):
        for i, v in enumerate(cfg):
            items.extend(flatten_cfg(v, f"{prefix}[{i}]"))
    else:
        items.append((prefix, cfg))
    return items


def try_hydra_instantiate(obj_cfg: dict):
    """
    Best-effort Hydra-style instantiation:
      obj_cfg = {"_target_": "pkg.module.ClassName", ...kwargs...}
    """
    if not isinstance(obj_cfg, dict):
        return None
    target = obj_cfg.get("_target_")
    if not target:
        return None
    mod_name, cls_name = target.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    kwargs = {k: v for k, v in obj_cfg.items() if not str(k).startswith("_")}
    return cls(**kwargs)


def build_models_best_effort(cfg: dict, teacher_ckpt: dict, student_ckpt: dict, device: str):
    """
    This tries to build modules only if the config provides enough info.
    If it can’t, we return {} and still generate a useful PDF from checkpoints.
    """
    built = {}

    # Try Hydra-like config blocks first
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    student_cfg = cfg.get("student", {}) if isinstance(cfg, dict) else {}

    # Candidates: tweak keys if your config differs
    candidates = [
        ("vae", model_cfg.get("vae")),
        ("idenc", model_cfg.get("idenc") or model_cfg.get("id_encoder") or model_cfg.get("identity_encoder")),
        ("denoiser", model_cfg.get("denoiser") or model_cfg.get("unet") or model_cfg.get("diffusion")),
        ("student_denoiser", student_cfg.get("denoiser") or student_cfg.get("student_denoiser") or student_cfg.get("unet")),
    ]

    for name, block in candidates:
        try:
            m = try_hydra_instantiate(block)
            if m is None:
                continue
            m.to(device)
            m.eval()

            # Load weights if present in ckpts
            sd = None
            if name in teacher_ckpt:
                sd = teacher_ckpt.get(name)
            if name in student_ckpt:
                sd = student_ckpt.get(name)
            # student checkpoint uses "student_denoiser"
            if name == "denoiser" and "denoiser" in teacher_ckpt:
                sd = teacher_ckpt["denoiser"]
            if name == "student_denoiser" and "student_denoiser" in student_ckpt:
                sd = student_ckpt["student_denoiser"]

            if isinstance(sd, dict):
                missing, unexpected = m.load_state_dict(sd, strict=False)
                built[name] = (m, missing, unexpected)
            else:
                built[name] = (m, [], [])
        except Exception:
            continue

    return built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="path to teacher.pt")
    ap.add_argument("--student", required=True, help="path to student.pt")
    ap.add_argument("--out", required=True, help="output pdf path")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--max_tensors", type=int, default=250, help="max tensor rows to print per component")
    args = ap.parse_args()

    teacher_path = Path(args.teacher)
    student_path = Path(args.student)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    teacher_ckpt = torch.load(str(teacher_path), map_location="cpu")
    student_ckpt = torch.load(str(student_path), map_location="cpu")

    # cfg priority: student.cfg (often more complete) else teacher.cfg
    cfg = student_ckpt.get("cfg") or teacher_ckpt.get("cfg") or {}
    teacher_cfg = student_ckpt.get("teacher_cfg") or teacher_ckpt.get("cfg") or {}

    styles = getSampleStyleSheet()
    story = []

    def H(txt):
        story.append(Paragraph(txt, styles["Heading2"]))
        story.append(Spacer(1, 8))

    def P(txt):
        story.append(Paragraph(txt, styles["BodyText"]))
        story.append(Spacer(1, 6))

    def PRE(txt):
        story.append(Preformatted(txt, styles["Code"]))
        story.append(Spacer(1, 8))

    # --- Title ---
    story.append(Paragraph("TempoSyncDiff — Internal Architecture / Network Design", styles["Title"]))
    story.append(Spacer(1, 10))
    P(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    P(f"Teacher checkpoint: {teacher_path} ({human_bytes(teacher_path.stat().st_size)})")
    P(f"Student checkpoint: {student_path} ({human_bytes(student_path.stat().st_size)})")
    P(f"Device requested: {args.device} | Using: {device}")
    story.append(PageBreak())

    # --- High-level pipeline diagram ---
    H("1) System / Training Pipeline (Network Design)")
    PRE(textwrap.dedent("""\
    [LRS3 dataset] 
        │
        ├─► Teacher training (VAE + IdentityEncoder + Denoiser)
        │       └─ early stopping → results/checkpoints/teacher.pt
        │
        ├─► Student distillation (StudentDenoiser learns from Teacher)
        │       └─ early stopping → results/checkpoints/student.pt
        │
        ├─► Evaluation (scripts/evaluate.py)
        │       └─ results/eval_student/metrics/eval.json
        │
        └─► Inference / sampling (scripts/inference_realtime.py)
                └─ output video (mp4) / frames
    """))
    story.append(PageBreak())

    # --- Checkpoint structure ---
    H("2) Checkpoint Structure")
    PRE("Teacher keys:\n  " + ", ".join(list(teacher_ckpt.keys())))
    PRE("Student keys:\n  " + ", ".join(list(student_ckpt.keys())))

    # --- Parameter counts table ---
    H("3) Parameter Counts (from state_dict tensors)")
    rows = [["Component", "Checkpoint", "Tensors", "Parameters"]]

    for comp in ["vae", "idenc", "denoiser"]:
        sd = teacher_ckpt.get(comp, {})
        t, p = state_dict_stats(sd)
        rows.append([comp, "teacher", str(t), f"{p:,}"])

    for comp in ["vae", "idenc", "student_denoiser"]:
        sd = student_ckpt.get(comp, {})
        t, p = state_dict_stats(sd)
        rows.append([comp, "student", str(t), f"{p:,}"])

    tbl = Table(rows, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",(0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("VALIGN",(0,0),(-1,-1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # --- Config summary ---
    H("4) Training / Distillation Config Summary (from ckpt['cfg'])")
    if isinstance(cfg, dict):
        # Show key highlights
        lr = None
        try:
            lr = cfg.get("train", {}).get("lr", None)
        except Exception:
            lr = None
        P(f"Detected train.lr: {lr}" if lr is not None else "train.lr not found in cfg.")
        # Print flattened config (trim)
        flat = flatten_cfg(cfg)
        lines = []
        for k, v in flat:
            s = str(v)
            if len(s) > 160:
                s = s[:160] + "…"
            lines.append(f"{k}: {s}")
        PRE("\n".join(lines[:500]))
    else:
        PRE("cfg is not a dict; printing str(cfg):\n" + str(cfg)[:8000])

    story.append(PageBreak())

    # --- Tensor listing ---
    H("5) Tensor Shapes (state_dict listing)")
    def list_tensors(title, sd: dict):
        lines = [title]
        if not isinstance(sd, dict):
            lines.append("  (not a dict)")
            return "\n".join(lines)

        for i, (k, v) in enumerate(sd.items()):
            if i >= args.max_tensors:
                lines.append(f"  ... (trimmed at {args.max_tensors} tensors)")
                break
            if torch.is_tensor(v):
                lines.append(f"  {k:60s}  {tuple(v.shape)}  dtype={str(v.dtype)}")
            else:
                lines.append(f"  {k:60s}  (non-tensor)")
        return "\n".join(lines)

    PRE(list_tensors("Teacher/vae:", teacher_ckpt.get("vae", {})))
    PRE(list_tensors("Teacher/idenc:", teacher_ckpt.get("idenc", {})))
    PRE(list_tensors("Teacher/denoiser:", teacher_ckpt.get("denoiser", {})))
    PRE(list_tensors("Student/student_denoiser:", student_ckpt.get("student_denoiser", {})))
    story.append(PageBreak())

    # --- Best-effort layer summaries (if buildable) ---
    H("6) Layer-by-layer Architecture (best-effort)")
    P("Note: Full layer summaries require instantiating the model classes. If your config uses Hydra-style '_target_' blocks, this section will auto-populate. Otherwise you still get full checkpoint + tensor-level details above.")

    built = build_models_best_effort(cfg, teacher_ckpt, student_ckpt, device=device)

    if not built:
        PRE("Could not auto-instantiate models from config.\n\n"
            "If you want this section filled, ensure your YAML has blocks like:\n"
            "model:\n"
            "  vae:\n"
            "    _target_: temposyncdiff.models.vae.VAE\n"
            "    ...kwargs...\n")
    else:
        for name, (m, missing, unexpected) in built.items():
            story.append(Paragraph(f"{name}", styles["Heading3"]))
            story.append(Spacer(1, 6))
            PRE(str(m)[:12000])

            if _HAS_TORCHINFO:
                try:
                    # We can't guess input shapes safely; torchinfo can still print a structural summary without input
                    # by using "verbose=0" and only module tree via repr above.
                    PRE("(torchinfo summary skipped: input shapes unknown for this repo)\n")
                except Exception as e:
                    PRE(f"(torchinfo failed: {e})")

            if missing or unexpected:
                PRE("load_state_dict(strict=False) notes:\n"
                    + (("missing keys:\n  " + "\n  ".join(missing) + "\n") if missing else "")
                    + (("unexpected keys:\n  " + "\n  ".join(unexpected) + "\n") if unexpected else "")
                )
            story.append(Spacer(1, 8))

    # Build PDF
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    print("WROTE PDF:", str(out_path))


if __name__ == "__main__":
    main()
