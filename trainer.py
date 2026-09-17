#!/usr/bin/env python
"""Self-contained LoRA SFT for Qwen3.5/3.6 (unified multimodal) on FOCUS QA -- no tools.

Deliberately does not use ms-swift / LLaMA-Factory: Qwen3.5 is new enough to need transformers
from main and the AutoModelForMultimodalLM class, and we do not want to fall back to Qwen3.

--calibrate-steps N runs only N optimiser steps and prints the projected time for one full
epoch, so the epoch cost is known before committing to a full run.

Speed knobs (all measured on the 27B, see scripts/hpc/README.md):
  * the `fla` package: without it transformers runs the 48 Gated-DeltaNet layers through a
    pure-torch chunked loop (63-step Python loop + one op-group per 64-token chunk, in forward,
    recompute AND backward) -- measured 37 ms fwd / 896 ms bwd per layer at 12k tokens vs
    2.4 / 6.2 ms with the Triton kernel. DISABLE_FLA=1 forces the torch path (fallback rung).
  * --ckpt-every-n K: gradient-checkpoint only every K-th decoder layer (K=1 = every layer).
    Recompute is a third of this LoRA run's FLOPs; K>1 trades activation memory for it.
  * --fp8: FP8 GEMMs for the frozen base linears (fp8_frozen.py), no grad_weight GEMMs.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

if os.environ.get("DISABLE_FLA") == "1":
    # transformers binds the gated-delta kernel at import: `importlib.import_module("fla")`
    # succeeds -> Triton kernel, raises -> pure-torch chunked loop. A None entry makes it raise.
    sys.modules["fla"] = None  # type: ignore[assignment]

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import (AutoModelForMultimodalLM, AutoProcessor, Trainer,  # noqa: E402
                          TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

# ----------------------------------------------------------------------------------------------
# sequence-length estimate from the real frame geometry (used for batching and for projections)
# ----------------------------------------------------------------------------------------------
PATCH, MERGE = 16, 2                      # Qwen3.5 vision: 16 px patches, 2x2 spatial merge
FACTOR = PATCH * MERGE                    # one LLM token covers a 32x32 px block
MIN_PIXELS = 4 * FACTOR * FACTOR
PER_FRAME_TEXT = 11                       # "[hh:mm:ss] " + vision start/end + newline
BASE_TEXT = 110                           # system prompt + template + question + answer
OLD_SYSTEM_TOKENS = 42                    # the one-line prompt BASE_TEXT was calibrated with
_SIZE_CACHE: dict[str, tuple[int, int]] = {}
_DEFAULT_SIZE = (768, 432)                # 107 of 119 videos; the rest are 720x576/720x480/640x360


def smart_resize(height: int, width: int, max_pixels: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS) -> tuple[int, int]:
    """Qwen2/3-VL image-processor resize rule: round to the 32-px grid, shrink to <= max_pixels
    (floor), never upscale beyond min_pixels. Frames are 0.23-0.41 MP, so above ~331k pixels
    most frames run at native size and extra max_pixels buys nothing."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def frame_tokens(path: str, max_pixels: int) -> int:
    """LLM tokens one frame becomes. Frame size is read once per video directory (all frames
    of a video share it); a missing file falls back to the dominant 768x432."""
    d = os.path.dirname(path)
    size = _SIZE_CACHE.get(d)
    if size is None:
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception:  # noqa: BLE001
            size = _DEFAULT_SIZE
        _SIZE_CACHE[d] = size
    w, h = size
    hb, wb = smart_resize(h, w, max_pixels)
    return (hb // FACTOR) * (wb // FACTOR)


def estimate_len(row, max_pixels: int = 200704, base_text: int = BASE_TEXT) -> int:
    """Sequence length of a row in real tokens (validated against the processor: within ~2%).
    The previous proxy assumed 256 tokens/frame from a 14-px patch; the model uses 16-px
    patches, so 200704 px is ~180-195 tokens and a 16384 'budget' was really ~12.6k tokens."""
    imgs = row.get("images", [])
    if not imgs:
        return base_text
    per = frame_tokens(imgs[0], max_pixels)   # one video per row -> one size
    return len(imgs) * (per + PER_FRAME_TEXT) + base_text


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    """Variable-size batches holding <= `budget` PADDED tokens, grouped by length.

    Half of this dataset is <=10 frames while the tail runs to 121, so a fixed batch size of 1
    spends most of its forward/backward passes on tiny sequences at negligible GPU occupancy.
    Packing length-similar rows to a token budget cuts the number of passes ~2.5x at 16k.

    Shuffle -> mega-chunk -> sort inside the chunk -> fill batches -> shuffle batch order, so
    rows still change partners every epoch instead of being frozen into a sorted order.
    A row longer than the budget becomes its own batch; nothing is ever dropped.
    """

    def __init__(self, lengths, budget, mega_factor=256, seed=0):
        self.lengths = list(lengths)
        self.budget = int(budget)
        self.mega = int(mega_factor)
        self.seed = int(seed)
        self.epoch = 0
        self._batches = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._batches = None

    def _build(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=g).tolist()
        batches, longest = [], 0
        for start in range(0, len(order), self.mega):
            chunk = sorted(order[start:start + self.mega], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                nxt = max(cur_max, self.lengths[i])
                if cur and nxt * (len(cur) + 1) > self.budget:
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = nxt
            if cur:
                batches.append(cur)
        perm = torch.randperm(len(batches), generator=g).tolist()
        self._batches = [batches[i] for i in perm]
        if int(os.environ.get("RANK", "0")) == 0:
            sizes = [len(b) for b in self._batches]
            print(f"[batching] {len(self.lengths)} rows -> {len(self._batches)} batches "
                  f"(budget {self.budget} tok, mean {sum(sizes)/len(sizes):.2f} rows/batch, "
                  f"max {max(sizes)}), {len(self.lengths)/len(self._batches):.2f}x fewer passes")

    def __iter__(self):
        if self._batches is None:
            self._build()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._build()
        return len(self._batches)


class CkptPolicy:
    """Per-micro-batch gradient-checkpointing density. A static 'every N-th layer' cannot fit:
    the longest rows (121 frames, 23k tokens) need every layer checkpointed on 141 GB, while
    the median batch (<=10k tokens) leaves half the card idle. Rules "<max_tokens>:<every_n>"
    (every_n 0 = no checkpointing at all) are tried in order of max_tokens; a batch above the
    last limit gets every layer checkpointed. Toggling the per-layer flag between steps is
    safe with non-reentrant checkpointing (no static graph)."""

    def __init__(self, model, spec: str):
        self.rules = sorted((int(a), int(b)) for a, b in
                            (r.split(":") for r in spec.split(",") if r.strip()))
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = [m for n, m in base.named_modules() if "DecoderLayer" in type(m).__name__]
        if not self.layers:
            self.layers = [m for n, m in base.named_modules()
                           if hasattr(m, "gradient_checkpointing") and n.split(".")[-1].isdigit()
                           and "visual" not in n and "vision" not in n]
        self.cur = None

    def every_n_for(self, n_tokens: int) -> int:
        for lim, every in self.rules:
            if n_tokens <= lim:
                return every
        return 1

    def apply(self, n_tokens: int) -> int:
        k = self.every_n_for(n_tokens)
        if k != self.cur:
            for i, layer in enumerate(self.layers):
                layer.gradient_checkpointing = (k != 0) and (i % k == 0)
            self.cur = k
        return k


class AnswerLossTrainer(Trainer):
    """Cross-entropy on the supervised (answer) positions only, upcast AFTER selection.

    The stock loss does `logits.float()` on the full [seq, vocab] tensor -- ~8GB for a
    9k-token multimodal sequence with a 150k vocab, which OOMs the 32GB card. Only the
    handful of answer tokens carry labels, so selecting them first shrinks the upcast to
    megabytes with identical gradients.
    """

    LOGITS_TAIL = 192   # covers the longest answers (multi-timestamp lists) + chat suffix
    batch_sampler_obj = None    # set to a TokenBudgetBatchSampler to enable token batching
    ckpt_policy = None          # set to a CkptPolicy for per-batch checkpoint density

    def get_train_dataloader(self):
        if self.batch_sampler_obj is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader
        # NOTE: a plain (non-distributed) batch sampler on purpose -- accelerate's
        # BatchSamplerShard then hands every Nth batch to rank N and pads the tail so all
        # ranks run the SAME number of steps, which DDP requires or it deadlocks.
        dl = DataLoader(self.train_dataset, batch_sampler=self.batch_sampler_obj,
                        collate_fn=self.data_collator,
                        num_workers=self.args.dataloader_num_workers,
                        pin_memory=self.args.dataloader_pin_memory)
        return self.accelerator.prepare(dl)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if self.ckpt_policy is not None:
            self.ckpt_policy.apply(int(inputs["input_ids"].numel()))
        # Selecting after forward is NOT enough: accelerate's bf16 wrapper upcasts the whole
        # output to fp32 before this method runs. The logits must be small AT THE SOURCE, so
        # ask the model to emit only the tail window (answers are always at the end).
        # How many trailing positions must the model emit logits for? A FIXED 192-token tail is
        # only safe at batch size 1. With right-padded multi-row batches a short row's answer
        # sits BEFORE the padding, so a fixed tail can miss it entirely and that row would
        # silently contribute no loss. Take the window that covers every supervised token in
        # the batch (length grouping keeps this close to 192).
        sup = labels != -100
        keep = int(labels.shape[1] - int(torch.argmax(sup.int(), dim=1).min()))
        keep = max(min(keep, labels.shape[1]), self.LOGITS_TAIL)
        if not hasattr(self, "_ltk_ok"):
            try:
                out = model(**inputs, logits_to_keep=keep)
                self._ltk_ok = True
            except TypeError:
                self._ltk_ok = False
                out = model(**inputs)
        else:
            out = (model(**inputs, logits_to_keep=keep) if self._ltk_ok
                   else model(**inputs))
        logits = out.logits
        if logits.shape[1] < labels.shape[1]:
            labels = labels[:, -logits.shape[1]:]      # align labels to the kept tail
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if int(mask.sum()) == 0:
            raise RuntimeError("no supervised tokens inside the kept logits tail -- "
                               "answer longer than LOGITS_TAIL or padding exceeded it")
        loss = F.cross_entropy(shift_logits[mask].float(), shift_labels[mask])
        return (loss, out) if return_outputs else loss


_TS_RE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\]")
# the data builder's own rule for which questions get the fine grid (prepare_data.DENSE_RE):
# clock times, durations and percentages, whose +/-5 s metric punishes a coarse grid
_DENSE_RE = re.compile(r"hh:mm:ss|how long|\bduration\b|\bpercentage\b", re.I)


def is_dense(row) -> bool:
    return bool(_DENSE_RE.search(row["messages"][1]["content"]))


def parse_target_dt(spec) -> tuple[float, float]:
    """'2.5' -> (2.5, 2.5) uniform; '2.5/5' -> (2.5 dense, 5 normal), which is the grid
    data_v2 was built on and the only one with a measured holdout score."""
    if spec in (None, "", 0, 0.0):
        return (0.0, 0.0)
    t = str(spec).split("/")
    return (float(t[0]), float(t[-1]))


def row_dt(row) -> float:
    """Seconds between consecutive frames of a row, read from its own '[hh:mm:ss]' markers
    (median, robust to the last interval being short). 0 if the row has fewer than 2."""
    t = [int(a) * 3600 + int(b) * 60 + int(c)
         for a, b, c in _TS_RE.findall(row["messages"][1]["content"])]
    if len(t) < 2:
        return 0.0
    gaps = sorted(b - a for a, b in zip(t, t[1:]))
    return float(gaps[len(gaps) // 2])


def thin_row(row, stride: int = 1, max_frames: int = 0, target_dt=0.0):
    """Runtime DENSITY knob: keep every `stride`-th frame (always the first and the last) and
    drop the matching '[hh:mm:ss] <image>' segments from the user turn, so a jsonl rendered on
    a fine grid can be trained on a coarser one without re-extracting frames.

    `target_dt` (seconds) is the preferred knob: each row is thinned to the smallest stride
    whose resulting spacing is at least target_dt, so ONE rendering serves every grid coarser
    than itself. It may be a single number (uniform) or "<dense>/<normal>" -- the two question
    classes start at different spacings, so a uniform stride cannot express a mixed grid. From
    a 1.25 s (dense) / 2.5 s (normal) rendering: "2.5" = the unified 2.5 s grid, "2.5/5" =
    exactly the data_v2 grid that scored 70.2% on the holdout.

    `max_frames` additionally caps a row (memory bound for the longest ones).
    Rows with <= 2 frames are returned unchanged."""
    imgs = row.get("images", [])
    n = len(imgs)
    if n <= 2:
        return row
    k = max(1, int(stride))
    if target_dt:
        td_dense, td_norm = parse_target_dt(target_dt)
        td = td_dense if is_dense(row) else td_norm
        dt = row_dt(row)
        if dt > 0 and td > 0:
            k = max(k, int(round(td / dt)) or 1)
    if max_frames and n > max_frames:
        k = max(k, math.ceil((n - 1) / max(max_frames - 1, 1)))
    if k <= 1:
        return row
    keep = sorted(set(range(0, n, k)) | {n - 1})
    sys_m, user_m, asst_m = row["messages"]
    parts = user_m["content"].split("<image>")        # parts[i] precedes image i; parts[-1] = question
    if len(parts) != n + 1:                           # unexpected layout: leave the row alone
        return row
    new_user = dict(user_m)
    new_user["content"] = "<image>".join([parts[i] for i in keep] + [parts[-1]])
    out = dict(row)
    out["images"] = [imgs[i] for i in keep]
    out["messages"] = [sys_m, new_user, asst_m]
    return out


class FocusSFT(Dataset):
    def __init__(self, path: Path, frame_stride: int = 1, max_frames: int = 0,
                 target_dt=0.0, system_prompt=None, procedure_map: dict | None = None):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
        self.frame_stride, self.max_frames = int(frame_stride), int(max_frames)
        # argparse hands us a STRING ("0", "2.5", "2.5/5"); "0" is truthy, so normalise here or
        # every row is parsed for timestamps even with thinning switched off
        self.target_dt = target_dt if any(parse_target_dt(target_dt)) else 0.0
        self.thins = self.frame_stride > 1 or bool(self.max_frames) or bool(self.target_dt)
        self.system_prompt = system_prompt        # fo_prompt.SystemPrompt, or None to keep the
        self.procedure_map = procedure_map or {}  # system message the jsonl was built with

    def __len__(self):
        return len(self.rows)

    def procedure_of(self, row):
        """Prefer the row's own field (v05 carries it); fall back to a uid -> procedure map
        joined from the ground truth, so files built before this change still work."""
        p = row.get("procedure_type")
        if p:
            return p
        uid = str(row.get("uid", ""))
        return self.procedure_map.get(uid.split("@", 1)[0])

    def __getitem__(self, i):
        r = self.rows[i]
        if self.thins:
            r = thin_row(r, self.frame_stride, self.max_frames, self.target_dt)
        if self.system_prompt is not None:
            # seed=i: every example sees a different class ORDER, fixed across epochs and
            # dataloader workers, so the order cannot be learned as if it carried meaning
            r = self.system_prompt.apply(r, seed=i, procedure=self.procedure_of(r))
        return r


def load_procedure_map(gt_csv) -> dict:
    """id -> procedure_type from the ground-truth CSV, for jsonl files that predate the field."""
    try:
        import csv
        with open(gt_csv, newline="") as fh:
            return {row["id"]: row["procedure_type"] for row in csv.DictReader(fh)
                    if row.get("procedure_type")}
    except Exception as e:  # noqa: BLE001
        print(f"[fo] could not read {gt_csv} ({type(e).__name__}); procedure type omitted")
        return {}


def build_messages(row, with_answer: bool):
    """Interleave text and images exactly as the jsonl lays them out.

    The user content is '[hh:mm:ss] <image>\n... question'; each <image> token is replaced by
    the next image IN PLACE so every frame keeps its preceding timestamp. Clumping the images
    before the text (the naive layout) silently detaches all timestamps from their frames.
    """
    sys_m, user_m, asst_m = row["messages"]
    parts = user_m["content"].split("<image>")
    imgs = row["images"]
    content = []
    for i, chunk in enumerate(parts):
        if chunk.strip():
            content.append({"type": "text", "text": chunk.strip()})
        if i < len(parts) - 1 and i < len(imgs):
            content.append({"type": "image", "image": imgs[i]})
    msgs = [{"role": "system", "content": sys_m["content"]},
            {"role": "user", "content": content}]
    if with_answer:
        msgs.append({"role": "assistant", "content": asst_m["content"]})
    return msgs


class Collator:
    """Render each example, mask the prompt so loss is only on the assistant answer."""
    def __init__(self, processor, max_len, two_pass: bool = False):
        self.p = processor
        self.max_len = max_len
        self.two_pass = two_pass

    def _images(self, row):
        return [Image.open(p).convert("RGB") for p in row["images"]]

    def prompt_len(self, full: str, prompt: str, imgs, enc, n_full: int) -> int:
        """Number of prompt tokens (everything before the answer) in the rendered sequence.

        The reference way runs the processor a SECOND time on prompt+images just to count
        tokens -- that doubles the CPU image work per row (decode, resize, normalise, 121
        frames in the long rows). The image tokens are identical in both renderings and the
        prompt is a token-level prefix of the full text, so the difference between the two
        TEXT-ONLY tokenisations (placeholders unexpanded) equals the difference between the
        expanded ones. test_collator.py asserts equality with the two-pass count."""
        if self.two_pass:
            return self.p(text=[prompt], images=imgs, return_tensors="pt")["input_ids"].shape[1]
        tok = self.p.tokenizer
        n_full_text = len(tok(full, add_special_tokens=False)["input_ids"])
        n_prompt_text = len(tok(prompt, add_special_tokens=False)["input_ids"])
        return n_full - (n_full_text - n_prompt_text)

    def __call__(self, batch):
        input_ids, labels_list = [], []
        feats, extras = [], []
        for row in batch:
            imgs = self._images(row)
            full = self.p.apply_chat_template(build_messages(row, True), tokenize=False,
                                              add_generation_prompt=False)
            prompt = self.p.apply_chat_template(build_messages(row, False), tokenize=False,
                                                add_generation_prompt=True)
            enc = self.p(text=[full], images=imgs, return_tensors="pt")
            # NEVER truncate: chopping input_ids while keeping all pixel features desyncs
            # image placeholders from vision features and the model refuses the batch.
            # Sequence length is bounded by construction (frame cap x tokens/frame + text).
            ids = enc["input_ids"][0]
            if len(ids) > self.max_len:
                print(f"warn: sample len {len(ids)} > max_len {self.max_len}; kept whole")
            n_prompt = self.prompt_len(full, prompt, imgs, enc, len(ids))
            lab = ids.clone()
            lab[:min(n_prompt, len(lab))] = -100          # supervise only the answer
            input_ids.append(ids)
            labels_list.append(lab)
            # Vision tensors (pixel_values [patches, d], image_grid_thw [n_img, 3]) carry NO
            # batch dim -- they are flattened across images. Keep them whole; concatenating
            # along dim 0 across samples is the correct batch merge. Indexing [0] here ships
            # one patch row / a 1-D grid and the model dies on grid_thw[:, 0].
            # Two DIFFERENT kinds of extra tensor come back and they must be merged
            # differently once a batch holds more than one row:
            #   * sequence-aligned, shape [1, seq_len] (e.g. mm_token_type_ids, which marks the
            #     image-placeholder positions) -> pad like input_ids, then stack.
            #   * vision tensors carrying no batch dim (pixel_values [patches, d],
            #     image_grid_thw [n_img, 3]) -> concatenate along dim 0.
            # Concatenating the first kind blows up as soon as two rows differ in length
            # ("Sizes of tensors must match except in dimension 0"), which never showed at
            # batch size 1 because a single row is never concatenated with anything.
            feat, seq_extra = {}, {}
            for k, v in enc.items():
                if k in ("input_ids", "attention_mask") or not isinstance(v, torch.Tensor):
                    continue
                if v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == len(ids):
                    seq_extra[k] = v[0]
                else:
                    feat[k] = v
            feats.append(feat)
            extras.append(seq_extra)
        # right-pad to the longest in batch
        pad_id = self.p.tokenizer.pad_token_id or self.p.tokenizer.eos_token_id
        maxn = max(len(x) for x in input_ids)
        batch_ids, batch_lab, attn = [], [], []
        for ids, lab in zip(input_ids, labels_list):
            padn = maxn - len(ids)
            batch_ids.append(torch.cat([ids, torch.full((padn,), pad_id)]))
            batch_lab.append(torch.cat([lab, torch.full((padn,), -100)]))
            attn.append(torch.cat([torch.ones(len(ids)), torch.zeros(padn)]))
        out = {"input_ids": torch.stack(batch_ids).long(),
               "labels": torch.stack(batch_lab).long(),
               "attention_mask": torch.stack(attn).long()}
        # transformers' sdpa path: `is_causal = q_length > 1 and attention_mask is None`.
        # Passing a mask forces a non-flash kernel -- and at this model's head_dim=256 the
        # memory-efficient backend is UNSUPPORTED, so the fallback is the naive math kernel.
        # Dropping the mask is safe even for padded batches because padding above is on the
        # RIGHT and attention is causal: position i only ever sees positions <= i, so real
        # tokens never attend to trailing pads. The pad rows produce garbage logits, which the
        # loss discards (their labels are -100).
        if os.environ.get("NO_ATTN_MASK") == "1":
            out.pop("attention_mask")
        for k in feats[0]:
            out[k] = torch.cat([f[k] for f in feats if k in f], dim=0)
        for k in (extras[0] if extras else {}):
            padded = []
            for e in extras:
                x = e[k]
                padded.append(torch.cat([x, torch.zeros(maxn - len(x), dtype=x.dtype)]))
            out[k] = torch.stack(padded)
        return out


# ----------------------------------------------------------------------------------------------
# model setup shared by the trainer and the benchmark/probe
# ----------------------------------------------------------------------------------------------
def _set_partial_checkpointing(model, every_n: int):
    """Checkpoint only every `every_n`-th decoder layer. Uses transformers' own
    `every_n_layers` when the installed version has it, else toggles the per-layer flag."""
    sig = inspect.signature(model.gradient_checkpointing_enable).parameters
    kw = {"use_reentrant": False}
    if "every_n_layers" in sig:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw, every_n_layers=every_n)
        return "transformers every_n_layers"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
    if every_n > 1:
        layers = [m for n, m in model.named_modules()
                  if hasattr(m, "gradient_checkpointing") and n.split(".")[-1].isdigit()
                  and "visual" not in n and "vision" not in n]
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = (i % every_n == 0)
    return "manual per-layer flag"


def setup_model(model_name: str, max_pixels: int, lora_rank: int, four_bit: bool = False,
                ckpt_every_n: int = 1, fp8: bool = False, init_adapter: str | None = None,
                device=None, fp8_scope: str = "mlp"):
    processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels,
                                              trust_remote_code=True)
    model_kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
                    attn_implementation="sdpa")
    if four_bit:
        from transformers import BitsAndBytesConfig
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kw)
    if four_bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    # Non-reentrant checkpointing + input-require-grads: with the default reentrant mode and
    # frozen embeddings, checkpointing silently fails to engage and full per-layer
    # activations (~24GB at 9k tokens) stay resident -- observed as OOM on the 32GB card.
    how = _set_partial_checkpointing(model, ckpt_every_n)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[ckpt] gradient checkpointing every {ckpt_every_n} layer(s) ({how})")

    # LoRA on the language-side linear layers; freeze the vision tower to save memory.
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ("visual", "vision", "image_encoder")):
            p.requires_grad_(False)
    lora = LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    if init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        src = os.path.join(init_adapter, "adapter_model.safetensors")
        res = set_peft_model_state_dict(model, load_file(src))
        missing = getattr(res, "unexpected_keys", None)
        print(f"[init] LoRA initialised from {src}"
              + (f" (unexpected keys: {list(missing)[:3]})" if missing else ""))
    if device is not None:
        model.to(device)
    if fp8:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fp8_frozen import convert_frozen_linears_to_fp8
        if device is None:
            model.to(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
        n, freed = convert_frozen_linears_to_fp8(model, scope=fp8_scope)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[fp8] {n} frozen linears ({fp8_scope}) -> fp8 e4m3 rowwise, {freed:.1f} GiB released")
    return processor, model


def sdpa_context():
    """SDPA backend priority from $SDPA_BACKEND (flash|cudnn), else no-op.

    set_priority=True ORDERS the backends rather than disabling them, so any call the first
    choice cannot serve (e.g. the vision tower's non-causal attention) falls through instead
    of erroring. Measured 2026-09-03 on H200: flash vs cuDNN is a wash end to end."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    order = {"cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                       SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
             "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                       SDPBackend.MATH]}
    choice = os.environ.get("SDPA_BACKEND", "").lower()
    if choice in order:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[attn] SDPA backend priority: {choice} ({[b.name for b in order[choice]]})")
        return sdpa_kernel(order[choice], set_priority=True)
    return contextlib.nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-pixels", type=int, default=401408)
    ap.add_argument("--four-bit", action="store_true", help="QLoRA: 4-bit base (if you OOM)")
    ap.add_argument("--calibrate-steps", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-fraction", type=float, default=0.0,
                    help="save every FRACTION of an epoch (0.5 = twice per epoch), computed from "
                         "the batch sampler once the data config is known. Overrides "
                         "--save-steps, because steps/epoch depends on the grid the job picks.")
    ap.add_argument("--init-adapter", default=None,
                    help="directory of a finished LoRA adapter to START FROM. Used to train a "
                         "further epoch on top of a completed one WITHOUT rewinding to the last "
                         "periodic checkpoint (--resume-from would redo every step since it).")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="max PADDED tokens per micro-batch (REAL tokens, from frame geometry); "
                         "enables length-grouped variable batching (0 = fixed --batch-size)")
    ap.add_argument("--ckpt-every-n", type=int, default=1,
                    help="gradient-checkpoint every N-th decoder layer (1 = all layers)")
    ap.add_argument("--fp8", action="store_true",
                    help="FP8 GEMMs for the frozen base linears (see fp8_frozen.py)")
    ap.add_argument("--fp8-scope", default="mlp", choices=["mlp", "all"])
    ap.add_argument("--ckpt-policy", default="",
                    help="per-batch checkpoint density, e.g. '6000:0,12000:2,18000:3' = no "
                         "checkpointing up to 6k tokens, every 2nd layer up to 12k, every 3rd up "
                         "to 18k, every layer beyond (overrides --ckpt-every-n at run time)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="train on every k-th frame of each row (runtime density knob; the last "
                         "frame is always kept). 2 turns a 1.25 s grid into 2.5 s without re-rendering")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="thin any row above this many frames to fit (memory bound; 0 = off)")
    ap.add_argument("--target-dt", default="0",
                    help="thin each row to at least this many seconds between frames, from its "
                         "own timestamps (0 = off). '2.5' = uniform; '2.5/5' = dense/normal. "
                         "One fine rendering then serves every coarser grid.")
    ap.add_argument("--fo-definitions", default=None,
                    help="path to FO_definitions.json/.txt. The system prompt is then rebuilt "
                         "from it (general rule + class names + descriptions of the classes the "
                         "training data cannot teach) instead of the one baked into the jsonl. "
                         "The SAME builder runs in eval.py and the submission container.")
    ap.add_argument("--no-fo-prompt", action="store_true",
                    help="keep the jsonl's own system message (A/B against --fo-definitions)")
    ap.add_argument("--no-describe-unsupported", action="store_true",
                    help="list unsupported class names without their descriptions (172 vs 277 "
                         "system tokens); a class we have no examples of is then unrecognisable")
    ap.add_argument("--gt-csv", default="data/ground_truth/focus_ground_truth_all.csv",
                    help="only used to fill procedure_type for jsonl rows that lack it")
    ap.add_argument("--two-pass-collator", action="store_true",
                    help="count prompt tokens by re-running the processor (reference path)")
    ap.add_argument("--resume-from", default=None,
                    help="path to a checkpoint-N dir (or 'auto' for the newest in --output) to "
                         "continue a run that hit the walltime; optimiser/scheduler/data order "
                         "are restored, so epoch 2 continues exactly where epoch 1 stopped")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        import fla  # noqa: F401
        kern = "fla Triton kernels"
    except Exception:  # noqa: BLE001
        kern = "PURE-TORCH FALLBACK (slow: install fla-core)"
    if rank == 0:
        print(f"[kernels] gated-delta rule: {kern}")

    processor, model = setup_model(
        args.model, args.max_pixels, args.lora_rank, four_bit=args.four_bit,
        ckpt_every_n=args.ckpt_every_n, fp8=args.fp8, init_adapter=args.init_adapter,
        device=torch.device(f"cuda:{local_rank}") if args.fp8 else None, fp8_scope=args.fp8_scope)
    # Save the processor/tokenizer config (with the exact max_pixels this run trains at) to the
    # output dir IMMEDIATELY, before training starts -- not only at the end. Periodic Trainer
    # checkpoints (output/checkpoint-N/) save the LoRA adapter alone, so without this, a job
    # that times out, OOMs or gets preempted before trainer.train() returns leaves an adapter
    # with nothing to load it with, and reloading the processor from the base model afterwards
    # would silently pick up the base model's DEFAULT max_pixels instead of this run's.
    # Rank-0 only under torchrun DDP (every process would otherwise race on the same files).
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(str(args.output))
        print(f"[startup] processor config saved to {args.output} (max_pixels={args.max_pixels})")
        # A checkpoint is only usable if whoever loads it reproduces the PROMPT and the FRAME
        # GRID it was trained with; both are run-time choices here, so record them next to the
        # adapter and ship the builder itself. eval.py and the container then have one source.
        here = Path(__file__).resolve().parent
        for extra in ("fo_prompt.py", "FO_definitions.json"):
            src = here / extra
            if src.exists():
                shutil.copy(src, args.output / extra)
        (args.output / "RUN_CONFIG.json").write_text(json.dumps({
            "model": args.model, "lora_rank": args.lora_rank, "lr": args.lr,
            "max_pixels": args.max_pixels, "target_dt": args.target_dt,
            "frame_stride": args.frame_stride, "max_frames": args.max_frames,
            "token_budget": args.token_budget, "grad_accum": args.grad_accum,
            "epochs": args.epochs, "train": str(args.train),
            "fo_prompt": not args.no_fo_prompt,
            "fo_describe_unsupported": not args.no_describe_unsupported,
            "fp8": bool(args.fp8), "fp8_scope": args.fp8_scope,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "eval_hint": ("evaluate with sft_vlm/baseline/eval.py --max-pixels "
                          f"{args.max_pixels}" + ("" if args.no_fo_prompt else
                          " (the FO system prompt is REQUIRED: this checkpoint was trained "
                          "with it, and eval.py rebuilds it from FO_definitions.json here)")),
        }, indent=1))
        print(f"[startup] wrote RUN_CONFIG.json + the prompt contract to {args.output}")
    model.print_trainable_parameters()

    sysp, base_text = None, BASE_TEXT
    proc_map = {}
    if not args.no_fo_prompt:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fo_prompt import SystemPrompt
        sysp = SystemPrompt(args.fo_definitions,
                            describe_unsupported=not args.no_describe_unsupported,
                            shuffle=True)
        base_text = BASE_TEXT - OLD_SYSTEM_TOKENS + sysp.token_cost()
        if rank == 0:
            print(f"[fo] system prompt from {args.fo_definitions or 'the shipped copy'}: "
                  f"{sysp.token_cost()} tokens (+{sysp.token_cost() - OLD_SYSTEM_TOKENS} vs the "
                  f"old one-liner); classes without training support: {sysp.unsupported}")
            print(f"[fo] example prompt:\n{sysp.for_row('Laparoscopic Cholecystectomy', 0)}")
    train_ds = FocusSFT(args.train, args.frame_stride, args.max_frames, args.target_dt)
    val_ds = (FocusSFT(args.val, args.frame_stride, args.max_frames, args.target_dt)
              if args.val and args.val.exists() else None)
    if sysp is not None:
        if not any(r.get("procedure_type") for r in train_ds.rows[:200]):
            proc_map = load_procedure_map(args.gt_csv)
            if rank == 0:
                print(f"[fo] rows carry no procedure_type; joined {len(proc_map)} ids "
                      f"from {args.gt_csv}")
        for ds in (train_ds, val_ds):
            if ds is not None:
                ds.system_prompt, ds.procedure_map = sysp, proc_map
        if rank == 0:
            hit = sum(1 for r in train_ds.rows[:500] if train_ds.procedure_of(r))
            print(f"[fo] procedure_type resolved for {hit}/500 sampled train rows")
    if rank == 0 and train_ds.thins:
        n0 = sum(len(r["images"]) for r in train_ds.rows)
        kept = [len(thin_row(r, train_ds.frame_stride, train_ds.max_frames,
                             train_ds.target_dt)["images"]) for r in train_ds.rows]
        print(f"[thin] stride={args.frame_stride} max_frames={args.max_frames} "
              f"target_dt={args.target_dt}: {n0} -> {sum(kept)} frames "
              f"({sum(kept)/n0:.2f}x), longest row {max(kept)} frames")
    collator = Collator(processor, args.max_len, two_pass=args.two_pass_collator)

    # transformers v5 slimmed TrainingArguments (warmup_ratio is gone, more may follow):
    # keep only the kwargs this installed version actually accepts, and say what was dropped.
    # Build the batch sampler BEFORE TrainingArguments: --save-fraction needs steps-per-epoch,
    # which only the sampler knows, and passing save_steps in is safer than mutating
    # trainer.args afterwards and trusting the callback to re-read it.
    SEED = 42
    sampler, save_steps = None, args.save_steps
    if args.token_budget > 0:
        lens = [estimate_len(train_ds[i], args.max_pixels, base_text)
                for i in range(len(train_ds))]
        if rank == 0:
            print(f"[batching] est. {sum(lens)/1e6:.1f}M tokens/epoch, longest row {max(lens)} tok")
        sampler = TokenBudgetBatchSampler(lens, budget=args.token_budget, seed=SEED)
        if args.save_fraction > 0:
            world = int(os.environ.get("WORLD_SIZE", "1"))
            steps_ep = len(sampler) / (world * args.grad_accum)
            save_steps = max(1, int(round(steps_ep * args.save_fraction)))
            if rank == 0:
                print(f"[save] {steps_ep:.0f} optimiser steps per epoch -> save every "
                      f"{save_steps} steps ({args.save_fraction:g} epoch), about "
                      f"{args.epochs / args.save_fraction:.0f} checkpoints over {args.epochs:g} "
                      f"epoch(s)")

    import dataclasses
    want = dict(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        bf16=True, logging_steps=5, save_steps=save_steps, save_total_limit=None,
        seed=SEED,
        dataloader_num_workers=int(os.environ.get("DL_WORKERS", "4")),
        report_to="none", remove_unused_columns=False,
        max_steps=args.calibrate_steps if args.calibrate_steps else -1,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # DDP across --nproc_per_node processes (torchrun): LoRA freezes the base model, so
        # only adapter params need gradient sync; unused-param scanning is unnecessary and slow.
        want["ddp_find_unused_parameters"] = False
    valid = {f.name for f in dataclasses.fields(TrainingArguments)}
    dropped = sorted(k for k in want if k not in valid)
    if dropped:
        print(f"note: this transformers version does not accept {dropped}; dropped")
    targs = TrainingArguments(**{k: v for k, v in want.items() if k in valid})
    trainer = AnswerLossTrainer(model=model, args=targs, train_dataset=train_ds,
                                eval_dataset=val_ds, data_collator=collator)
    if args.ckpt_policy:
        trainer.ckpt_policy = CkptPolicy(model, args.ckpt_policy)
        if rank == 0:
            print(f"[ckpt] adaptive policy {trainer.ckpt_policy.rules} over "
                  f"{len(trainer.ckpt_policy.layers)} decoder layers")
    if sampler is not None:
        trainer.batch_sampler_obj = sampler

    if os.environ.get("NO_ATTN_MASK") == "1" and rank == 0:
        print("[attn] attention_mask omitted for unpadded batches -> SDPA fast path "
              f"(flash backend available: {torch.backends.cuda.flash_sdp_enabled()})")

    resume = args.resume_from
    if resume == "auto":
        cks = sorted(args.output.glob("checkpoint-*"), key=lambda q: int(q.name.split("-")[1]))
        resume = str(cks[-1]) if cks else None
        print(f"[resume] auto-selected {resume or 'nothing (fresh start)'}")
    t0 = time.time()
    with sdpa_context():
        result = trainer.train(resume_from_checkpoint=resume)
    dt = time.time() - t0

    if args.calibrate_steps:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        sec_per_step = dt / max(args.calibrate_steps, 1)
        if trainer.batch_sampler_obj is not None:
            # Variable batch sizes: Trainer's train_samples_per_second assumes a FIXED
            # batch_size x grad_accum and badly under-reports token batching (it counted 2
            # rows/step when the sampler was really feeding ~19). Derive both numbers from
            # what actually runs: batches/epoch from the sampler, rows/step from its mean.
            n_batches = len(trainer.batch_sampler_obj)
            rows_per_batch = len(train_ds) / n_batches
            eff_batch = rows_per_batch * world * args.grad_accum
            steps_per_epoch = n_batches / (world * args.grad_accum)
        else:
            eff_batch = args.batch_size * args.grad_accum * world
            steps_per_epoch = len(train_ds) / eff_batch
        sps = eff_batch / sec_per_step
        est_h = steps_per_epoch * sec_per_step / 3600
        print("\n==================== EPOCH ESTIMATE ====================")
        print(f"  train samples          : {len(train_ds)}")
        print(f"  effective batch        : {eff_batch:.1f} rows/step  "
              f"(ga {args.grad_accum} x {world} gpu"
              f"{'' if trainer.batch_sampler_obj is None else f', ~{len(train_ds)/len(trainer.batch_sampler_obj):.2f} rows/batch'})")
        print(f"  seconds per step       : {sec_per_step:.1f}")
        print(f"  measured samples/sec   : {sps:.2f}   over {args.calibrate_steps} steps")
        print(f"  optimiser steps/epoch  : {steps_per_epoch:.0f}")
        print(f"  >> projected 1 epoch   : {est_h:.1f} h  ({est_h*60:.0f} min)")
        print("========================================================")
    else:
        trainer.save_model(str(args.output))
        processor.save_pretrained(str(args.output))
        print(f"done in {dt/3600:.2f} h -> {args.output}")


if __name__ == "__main__":
    main()
