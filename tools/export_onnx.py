import argparse
import inspect
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/checkpoints/student.pt")
    ap.add_argument("--out",  default="student_denoiser.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    sd  = ckpt["student_denoiser"]

    # ------------------------------------------------------------
    # IMPORTANT: Use the SAME import + constructor as distill_student.py
    # Example (YOU MUST EDIT THESE 2 LINES):
    #
    # from temposyncdiff.models.denoiser import StudentDenoiser
    # model = StudentDenoiser(**cfg["student"])
    # ------------------------------------------------------------
    from temposyncdiff.models.denoiser import StudentDenoiser  # <-- EDIT if different
    model = StudentDenoiser(**cfg["student"])                  # <-- EDIT if different

    model.load_state_dict(sd, strict=True)
    model.eval()

    print("[export] forward signature:", inspect.signature(model.forward))

    # ------------------------------------------------------------
    # Dummy inputs MUST match model.forward(...)
    # Best way: copy the exact student(...) call from scripts/evaluate.py
    # and use the same tensors/shapes.
    # ------------------------------------------------------------

    # TEMP PLACEHOLDER (you will adapt after checking forward signature)
    B = 1
    x = torch.randn(B, 4, 50, 28, 28)                 # e.g., noisy latent
    t = torch.randint(0, 1000, (B,), dtype=torch.long) # e.g., timestep
    cond = torch.randn(B, 256)                        # e.g., conditioning

    # Try exporting with 3 inputs first; adjust count/order to your forward()
    inputs = (x, t, cond)

    out_path = Path(args.out)
    torch.onnx.export(
        model,
        inputs,
        str(out_path),
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[f"in{i}" for i in range(len(inputs))],
        output_names=["out"],
        dynamic_axes={"in0": {0: "B"}, "out": {0: "B"}},
    )
    print("[export] wrote:", str(out_path))


if __name__ == "__main__":
    main()
