"""
FOCUS SEGMENT-track algorithm: finetuned Qwen3.6-27B, no tools.

The model is a LoRA-finetuned (merged) Qwen3.6-27B trained to map timestamped frames +
question -> answer in the grader's exact format. Everything below reproduces the TRAINING
protocol bit-for-bit -- sampling grid, timestamp strings, prompt, resolution, decoding.
Deviating from any of it silently costs points (measured during development: a mismatched
thinking-mode alone collapsed scores from ~42% to ~3%).

Protocol (must match training):
  * frames sampled on a uniform dt=2.5s grid over the clip, first AND last frame included,
    capped at 121 (dt widens if a clip would exceed the cap). This is the grid recorded in
    the checkpoint's RUN_CONFIG.json ("target_dt": "2.5"); the trainer logged 121 as the
    longest row it saw. The previous 9B submission used 5 s / 64 frames -- training and
    inference must agree or the temporal questions are answered off-grid;
  * every frame is preceded by the text `[hh:mm:ss]` -- the ORIGINAL video timeline, i.e.
    request.start_time + offset-in-clip (clips arrive pre-trimmed to the question window);
  * frames resized so the longest side is <= 768 px, processor max_pixels=200704 (the value
    in RUN_CONFIG.json; 768x432 sources render at 576x320, i.e. 180 tokens per frame);
  * system prompt built by fo_prompt.py from FO_definitions.json -- the general foreign-object
    rule, the class names, descriptions for the classes our training data cannot teach, and the
    request's procedure_type. The SAME builder produced the training prompts, so this must stay
    in lockstep with the checkpoint: a checkpoint trained without it must run with
    USE_FO_PROMPT=False. The definitions are read from /input when the platform mounts them, so
    a test-phase file with extra object classes flows straight into the prompt;
  * chat template with add_generation_prompt=True and enable_thinking=False (Qwen3.5 is a
    reasoning model; training used the empty-think pattern), anything before '</think>'
    stripped from the decode, greedy decoding, max_new_tokens=256.

Latency, measured on an H100 (80 GB) with this checkpoint merged, at 200704 px:
frames   7    25    49    61   121
time   0.5s  1.1s  2.5s  3.0s  5.8s     peak VRAM 51.6 -> 58.0 GiB, model load 41 s
The worst question on this grid (121 frames) uses 5.8 s of the 15 s per-question budget and
58 GiB of the card, so both have wide margins. A soft budget guard answers "" for remaining questions if the run ever
nears the pooled limit, since an overrun >20% forfeits the whole batch.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# decord must be imported AFTER torch: importing it first breaks CUDA
# initialisation (a known decord issue).
import decord
from PIL import Image
from focus import Request, Response, load_requests, save_items
from transformers import AutoModelForMultimodalLM, AutoProcessor

from fo_prompt import SystemPrompt

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESOURCES_PATH = Path(__file__).parent / "resources"
MODEL_PATH = RESOURCES_PATH / "qwen36_27b_v05"

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

# Plain clips: the model was trained with TEXT timestamps, not burned-in overlays.
VIDEO_DIR = INPUT_PATH / "plain"

# ── Training-protocol constants (do not tune independently of the checkpoint) ──
# Set False only for a checkpoint trained WITHOUT the definitions in its system prompt.
USE_FO_PROMPT = True
# The platform mounts the (possibly extended) definitions here; the copy shipped beside
# fo_prompt.py is the fallback.
FO_DEFINITIONS = INPUT_PATH / "FO_definitions.json"
LEGACY_SYSTEM = ("You answer questions about foreign objects (sponges, needles, clips, drains, "
                 "specimens, etc.) in surgical video. Answer concisely in exactly the format "
                 "the question asks for, and nothing else.")
# ALL FOUR come from the checkpoint's RUN_CONFIG.json -- do not tune them independently.
DT_S = 2.5           # "target_dt": "2.5" -- a uniform 2.5 s grid for EVERY question. The 9B
                     # submission used 5 s, and the 27B v05 run halved it precisely because a
                     # 5 s gap can miss a foreign object that is only visible for four seconds.
CAP_FRAMES = 121     # the longest row the trainer saw on this grid (299 s clip / 2.5 s)
MAX_SIDE = 768       # longest image side before the processor (as the frames were extracted)
MAX_PIXELS = 200704  # "max_pixels": 200704
MAX_NEW_TOKENS = 256

SETUP_BUDGET_S = 120.0
PER_QUESTION_BUDGET_S = 15.0
BUDGET_MARGIN_S = 10.0   # answer "" for the tail rather than risk the >20% forfeit


def hhmmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def log_environment(device: torch.device) -> None:
    log.info("--- Environment ---")
    log.info("  torch          : %s (built against CUDA %s)", torch.__version__, torch.version.cuda)
    arch = torch.cuda.get_arch_list() or (torch._C._cuda_getArchFlags() or "").split()
    log.info("  torch kernels  : %s", " ".join(arch) or "unknown")
    if device.type != "cuda":
        log.warning("  No GPU visible -- running on CPU. The platform always provides one.")
        return
    free, total = torch.cuda.mem_get_info(0)
    log.info("  GPU            : %s", torch.cuda.get_device_name(0))
    log.info("  VRAM           : %.1f GiB free of %.1f GiB", free / 1024**3, total / 1024**3)


def sample_clip_times(duration_s: float) -> list[float]:
    """Uniform dt=5s grid over the clip, first and last frame always included, capped.

    Identical to the training sampler: a fixed frame COUNT would make the step vary
    with clip length, and the sparse end makes +/-5s timestamp questions unanswerable.
    """
    dur = max(duration_s, 1e-3)
    dt = max(DT_S, dur / max(CAP_FRAMES - 1, 1))
    times = list(np.arange(0.0, dur, dt))
    if not times or dur - times[-1] > 1e-6:
        times.append(dur)
    return times[:CAP_FRAMES]


def decode_frames(video_path: Path, clip_times: list[float]) -> tuple[list[Image.Image], list[float]]:
    """Batched decord read of the requested clip times -> PIL images (longest side <=768)."""
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps()) or 5.0
    n = len(vr)
    idx, kept = [], []
    for t in clip_times:
        i = min(int(round(t * fps)), n - 1)
        if i >= 0:
            idx.append(i)
            kept.append(t)
    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8 RGB
    del vr
    images = []
    for arr in frames:
        img = Image.fromarray(arr)
        if max(img.size) > MAX_SIDE:
            s = MAX_SIDE / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)))
        images.append(img)
    return images, kept


def build_messages(req: Request, images: list[Image.Image], clip_times: list[float],
                   system: str) -> list[dict]:
    """The exact training layout: [hh:mm:ss] before EVERY frame, absolute timeline.

    The clip is pre-trimmed, so absolute time = request.start_time + clip offset. The
    stamps are interleaved IN PLACE -- clumping the images before the text detaches every
    timestamp from its frame and forfeits the temporal-grounding questions.
    """
    content = []
    for img, t in zip(images, clip_times):
        content.append({"type": "text", "text": f"[{hhmmss(req.start_time + t)}]"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": str(req.question)})
    return [{"role": "system", "content": system},
            {"role": "user", "content": content}]


def answer_one(model, processor, device: torch.device, req: Request, sysp) -> str:
    images, clip_times = decode_frames(clip_path_for(req), sample_clip_times(float(req.duration)))
    # seed=None: the class order is the file's, so the prompt is byte-reproducible. Training
    # shuffled the order per example precisely so that any single fixed order is in
    # distribution here.
    system = (sysp.for_row(getattr(req, "procedure_type", None), None)
              if sysp is not None else LEGACY_SYSTEM)
    messages = build_messages(req, images, clip_times, system)
    # enable_thinking=False: Qwen3.5 otherwise opens a think block and burns the whole
    # token budget on reasoning before any answer appears.
    try:
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = gen[0][inputs["input_ids"].shape[1]:]
    pred = processor.tokenizer.decode(new, skip_special_tokens=True).strip()
    if "</think>" in pred:                     # belt and braces: strip residual reasoning
        pred = pred.split("</think>", 1)[1].strip()
    # No valid FOCUS answer ends in a period (classes, integers, hh:mm:ss, yes/no), but the
    # official Binary/FOClass parsers reject "No." as a parse failure. Free insurance.
    return pred.rstrip(".").strip()


def clip_path_for(req: Request) -> Path:
    return VIDEO_DIR / f"{req.qID}.mp4"


def run() -> int:
    t_start = time.monotonic()
    log.info("=== FOCUS SEGMENT -- Qwen3.5-9B (no tools) -- inference start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log_environment(device)

    # ── Foreign-object definitions ────────────────────────────────────────────
    sysp = None
    if USE_FO_PROMPT:
        try:
            sysp = SystemPrompt(FO_DEFINITIONS if FO_DEFINITIONS.exists() else None)
            log.info("--- Foreign-object definitions ---")
            log.info("  source         : %s", FO_DEFINITIONS if FO_DEFINITIONS.exists()
                     else "the copy shipped in the image (platform mounted none)")
        except Exception:
            log.exception("could not build the FO system prompt; falling back to the legacy one")
            sysp = None
    if sysp is not None:
        from fo_prompt import parse_definitions
        _, classes = parse_definitions(sysp.text)
        log.info("  classes defined: %d (%s)", len(classes),
                 ", ".join(n for n, _ in classes))
        if sysp.unsupported:
            # The SDK warns that the test phase may add object types. Anything here had no (or
            # almost no) training support, so it is answerable only from its description in the
            # prompt -- worth seeing in the job log rather than discovering from a low score.
            log.warning("  NOT SUPPORTED BY TRAINING DATA: %s -- these carry their description "
                        "in the prompt and rely on zero-shot recognition",
                        ", ".join(sysp.unsupported))
        log.info("  system prompt  : %d chars", len(sysp.for_row("example", None)))

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    log.info("Batch of %d question(s)", len(requests))
    allowed = SETUP_BUDGET_S + PER_QUESTION_BUDGET_S * len(requests)

    # ── Model -- loaded ONCE for the whole batch ──────────────────────────────
    log.info("--- Loading model (once for the batch) from %s ---", MODEL_PATH)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), max_pixels=MAX_PIXELS,
                                              trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device)
    model.eval()
    log.info("Model loaded in %.1f s", time.monotonic() - t_start)
    if device.type == "cuda":
        log.info("  VRAM allocated: %.1f GiB", torch.cuda.memory_allocated(0) / 1024**3)

    # ── Inference -- one pass per question ────────────────────────────────────
    responses, n_failed, n_forfeited = [], 0, 0
    per_q: list[float] = []
    for i, req in enumerate(requests, start=1):
        t0 = time.monotonic()
        elapsed = t0 - t_start
        # Predictive guard: do not START a question the budget cannot afford. Checking
        # elapsed alone lets the final question blow through the limit (observed on the CPU
        # test: 238s spent against 165 allowed with the naive check). Estimate the next
        # question's cost from the worst of the last three.
        est = max(per_q[-3:]) if per_q else 8.0
        if elapsed + est > allowed - BUDGET_MARGIN_S:
            # Overrunning >20% forfeits the WHOLE batch; an empty answer only loses this
            # question. Never trade the batch for the tail.
            n_forfeited += 1
            responses.append(Response(qID=req.qID, content="", latency=0.0))
            continue
        try:
            answer = answer_one(model, processor, device, req, sysp)
        except Exception:
            n_failed += 1
            log.exception("[%d/%d] qID=%s failed; emitting empty answer", i, len(requests), req.qID)
            answer = ""
        latency = time.monotonic() - t0
        per_q.append(latency)
        responses.append(Response(qID=req.qID, content=answer, latency=latency))
        log.info("[%d/%d] qID=%s  %.2fs  %r", i, len(requests), req.qID, latency, answer[:80])

    log.info("Done: %d answered, %d failed, %d budget-guarded, %.1f s total (allowed %.0f s)",
             len(responses), n_failed, n_forfeited, time.monotonic() - t_start, allowed)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d response(s)", len(responses))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
