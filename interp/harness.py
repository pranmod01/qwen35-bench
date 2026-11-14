"""Prefix-cache patching for the Qwen3.5 hybrid (INTERP_PLAN.md, section 3).

A trial's prompt is split into a prefix (the study list through the opening of the test turn) and a probe (the
question, the options, the generation prompt and the answer prefill). The prefix is run once with a cache; the
cache then holds everything the probe can know about the list: the DeltaNet state M and conv state of every
linear-attention layer, and the keys/values of every full-attention layer. Patching is editing a fork of that
cache before the probe runs, or overriding a module's output at probe positions with a hook.

Nothing here changes the benchmark; the model is loaded through bench.models.Qwen35Evaluator and its prompt
rendering is reused so a patched trial sees the exact tokens the benchmark scored.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from bench.models import CHOICE_PREFILL

LINEAR, FULL = "linear_attention", "full_attention"


@dataclass
class Prompt:
    """A rendered trial: the full inputs, where the probe starts, and 3-row mRoPE position ids for every token."""
    inputs: dict
    split: int
    position_ids: torch.Tensor      # (3, 1, L)

    @property
    def length(self):
        return self.inputs["input_ids"].shape[1]


class Harness:
    def __init__(self, evaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.processor.tokenizer
        self.inner = self.model.model                      # Qwen3_5Model: .visual, .language_model
        self.layers = self.inner.language_model.layers
        self.layer_types = list(self.inner.language_model.config.layer_types)
        self.linear_layers = [i for i, t in enumerate(self.layer_types) if t == LINEAR]
        self.full_layers = [i for i, t in enumerate(self.layer_types) if t == FULL]
        self.vision_start_id = self.model.config.vision_start_token_id
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.choice_ids = {c: evaluator._choice_token_ids(c) for c in ("1", "2")}

    # ---- prompts -------------------------------------------------------------------------------------------

    def prepare(self, messages, prefill=CHOICE_PREFILL):
        inputs = self.ev._prepare_inputs(messages, prefill=prefill)
        inputs = dict(inputs.items()) if hasattr(inputs, "items") else dict(inputs)
        return Prompt(inputs=inputs, split=self.find_split(inputs["input_ids"]), position_ids=self.position_ids(inputs))

    def find_split(self, input_ids):
        """The probe starts after the last `<|im_start|>user\\n`, so the prefix ends on tokens every trial shares."""
        ids = input_ids[0].tolist()
        for i in reversed([k for k, t in enumerate(ids) if t == self.im_start_id]):
            for k in (1, 2, 3):
                if self.tokenizer.decode(ids[i + 1:i + 1 + k]) == "user\n":
                    return i + 1 + k
        raise ValueError("no user turn header found in the prompt")

    def position_ids(self, inputs):
        """Position ids for the whole prompt, computed once so the probe call continues the prefix exactly."""
        if inputs.get("image_grid_thw") is None:
            L = inputs["input_ids"].shape[1]
            return torch.arange(L, device=inputs["input_ids"].device).view(1, 1, -1).expand(3, 1, -1).contiguous()
        pos, _ = self.inner.get_rope_index(
            inputs["input_ids"], inputs["mm_token_type_ids"], image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get("attention_mask"),
        )
        return pos

    def part(self, prompt, which):
        """Inputs for the prefix or the probe: token fields sliced, images assigned by which part holds their tokens."""
        s, ids = prompt.split, prompt.inputs["input_ids"]
        sl = slice(None, s) if which == "prefix" else slice(s, None)
        out = {"input_ids": ids[:, sl], "position_ids": prompt.position_ids[:, :, sl]}
        if "mm_token_type_ids" in prompt.inputs:
            out["mm_token_type_ids"] = prompt.inputs["mm_token_type_ids"][:, sl]
        if prompt.inputs.get("pixel_values") is not None:
            grid = prompt.inputs["image_grid_thw"]
            patches = grid.prod(-1).tolist()
            k = int((ids[0, :s] == self.vision_start_id).sum())          # images whose tokens are in the prefix
            cut = sum(patches[:k])
            pv = prompt.inputs["pixel_values"]
            pv, grid = (pv[:cut], grid[:k]) if which == "prefix" else (pv[cut:], grid[k:])
            if len(grid):
                out["pixel_values"], out["image_grid_thw"] = pv, grid
        return out

    # ---- forwards ------------------------------------------------------------------------------------------

    @torch.no_grad()
    def full_pass(self, prompt):
        """Single-pass logits at the answer position, as the benchmark computes them (check 1's reference)."""
        inputs = {k: v for k, v in prompt.inputs.items() if k != "attention_mask"}
        return self.model(**inputs, position_ids=prompt.position_ids, logits_to_keep=1).logits[0, -1].float()

    @torch.no_grad()
    def run_prefix(self, prompt):
        """Run the prefix and return its cache: the model's entire memory of the study list."""
        out = self.model(**self.part(prompt, "prefix"), use_cache=True, logits_to_keep=1)
        return out.past_key_values

    @torch.no_grad()
    def run_probe(self, prompt, cache, hooks=()):
        """Logits at the answer position from the probe run on a fork of `cache` (the cache itself is not touched)."""
        fork = fork_cache(cache)
        with self.hooked(hooks):
            out = self.model(**self.part(prompt, "probe"), past_key_values=fork, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float()

    # ---- metric --------------------------------------------------------------------------------------------

    def logit_diff(self, logits, correct, counterfactual):
        """logsumexp over the token ids of the correct answer minus the same for the counterfactual answer."""
        def mass(ids):
            return torch.logsumexp(logits[list(ids)], dim=0)
        return float(mass(self.ids_for(correct)) - mass(self.ids_for(counterfactual)))

    def ids_for(self, answer):
        if isinstance(answer, str) and answer in self.choice_ids:
            return self.choice_ids[answer]
        return list(answer)

    # ---- module hooks (surfaces that are not in the cache) -------------------------------------------------

    def residual_module(self, layer):
        return self.layers[layer]

    def heads_module(self, layer):
        if self.layer_types[layer] != FULL:
            raise ValueError(f"layer {layer} is {self.layer_types[layer]}, not a full-attention layer")
        return self.layers[layer].self_attn.o_proj

    @contextmanager
    def hooked(self, hooks):
        handles = [h.attach(self) for h in hooks]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


# ---- cache forking and cache-surface patches ----------------------------------------------------------------

def fork_cache(cache):
    """A cache that can be run and patched without touching the original.

    Linear-attention layers are updated in place by the model (`copy_`), so their state dicts are cloned.
    Full-attention layers grow by `torch.cat`, which makes new tensors, so their keys/values are shared.
    """
    fork = copy.copy(cache)
    fork.layers = [fork_layer(layer) for layer in cache.layers]
    return fork


def fork_layer(layer):
    new = copy.copy(layer)
    if isinstance(layer, LinearAttentionCacheLayerMixin):
        clone = lambda d: {k: (v.clone() if v is not None else None) for k, v in d.items()}
        new.conv_states = clone(layer.conv_states)
        new.recurrent_states = clone(layer.recurrent_states)
        for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state",
                     "conv_kernel_size"):
            setattr(new, name, dict(getattr(layer, name)))
    return new


def copy_M(dst, src, layers):
    """Overwrite the DeltaNet state M of `layers` in `dst` with `src`'s (dst must be a fork)."""
    for i in layers:
        dst.layers[i].recurrent_states[0].copy_(src.layers[i].recurrent_states[0])


def copy_conv(dst, src, layers):
    """The short-convolution state: the last few projected inputs of a linear layer."""
    for i in layers:
        dst.layers[i].conv_states[0].copy_(src.layers[i].conv_states[0])


def copy_KV(dst, src, layers):
    """Point `dst`'s full-attention layers at `src`'s keys and values (shared, not copied; the probe cats onto them)."""
    for i in layers:
        dst.layers[i].keys = src.layers[i].keys
        dst.layers[i].values = src.layers[i].values


@dataclass
class CachePatch:
    """Which cache entries to take from `source` before the probe runs. surface: M | conv | KV."""
    surface: str
    layers: list
    source: object

    def apply(self, fork):
        {"M": copy_M, "conv": copy_conv, "KV": copy_KV}[self.surface](fork, self.source, self.layers)


def run_patched(harness, prompt, base, patches, hooks=()):
    """Probe logits from a fork of `base` with `patches` applied (each a CachePatch)."""
    fork = fork_cache(base)
    for patch in patches:
        patch.apply(fork)
    with harness.hooked(hooks):
        with torch.no_grad():
            out = harness.model(**harness.part(prompt, "probe"), past_key_values=fork, use_cache=True, logits_to_keep=1)
    return out.logits[0, -1].float()


# ---- hook surfaces: capture and override at probe positions -------------------------------------------------

class Capture:
    """Record a module's output (residual stream: decoder layer) or input (heads: o_proj) during a probe run."""

    def __init__(self, surface, layer):
        self.surface, self.layer, self.value = surface, layer, None

    def attach(self, harness):
        if self.surface == "residual":
            def hook(module, args, output):
                self.value = (output[0] if isinstance(output, tuple) else output).detach().clone()
            return harness.residual_module(self.layer).register_forward_hook(hook)
        if self.surface == "heads":
            def pre_hook(module, args):
                self.value = args[0].detach().clone()
            return harness.heads_module(self.layer).register_forward_pre_hook(pre_hook)
        raise ValueError(self.surface)


class Override:
    """Replace a slice of a module's output (residual) or input (heads) at probe positions with recorded values.

    positions: list of probe-relative token indices (negative counts from the end), or None for all.
    heads: list of head indices for the heads surface, or None for all.
    """

    def __init__(self, surface, layer, value, positions=None, heads=None, head_dim=None):
        self.surface, self.layer, self.value = surface, layer, value
        self.positions, self.heads, self.head_dim = positions, heads, head_dim

    def _apply(self, x):
        x = x.clone()
        pos = slice(None) if self.positions is None else torch.tensor(self.positions, device=x.device)
        if self.surface == "heads" and self.heads is not None:
            for h in self.heads:
                cols = slice(h * self.head_dim, (h + 1) * self.head_dim)
                x[:, pos, cols] = self.value[:, pos, cols].to(x.dtype)
        else:
            x[:, pos] = self.value[:, pos].to(x.dtype)
        return x

    def attach(self, harness):
        if self.surface == "residual":
            def hook(module, args, output):
                if isinstance(output, tuple):
                    return (self._apply(output[0]), *output[1:])
                return self._apply(output)
            return harness.residual_module(self.layer).register_forward_hook(hook)
        if self.surface == "heads":
            if self.head_dim is None:
                self.head_dim = harness.layers[self.layer].self_attn.head_dim
            def pre_hook(module, args):
                return (self._apply(args[0]), *args[1:])
            return harness.heads_module(self.layer).register_forward_pre_hook(pre_hook)
        raise ValueError(self.surface)
