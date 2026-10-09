"""Qwen3.5 evaluator, plus the layer-type override used for architecture variants."""
import hashlib
import re

import torch
from PIL import Image
from transformers import BatchFeature

from evaluators.base import BaseEvaluator

# Prefilled assistant turn for forced-choice scoring. The trailing space matters: after
# "Answer:" the model puts 99.7% of its mass on a space token and only ~3e-4 on each digit,
# which would leave the comparison in the tail of the distribution.
CHOICE_PREFILL = "Answer: "

# Stands in for a token item while the chat template renders; replaced by the item's IDs afterwards.
TOKENS_SENTINEL = "<<<tokens:{}>>>"
TOKENS_SENTINEL_RE = re.compile(r"<<<tokens:(\d+)>>>")

# Thinking-mode decoding when a config's `thinking` block leaves these out: Qwen's recommended
# settings for thinking mode. Greedy decoding is not offered as a default because Qwen warns it
# makes thinking traces loop.
THINKING_DEFAULTS = {"budget": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20}


def resolve_layer_types(base_layer_types, variant):
    """Return the layer_types list a variant asks for.

    variant accepts either an explicit `layer_types` list, or
    `full_attention_interval: n` (every nth layer full; 0 means all linear).
    """
    if not variant:
        return None

    if "layer_types" in variant:
        layer_types = list(variant["layer_types"])
        if len(layer_types) != len(base_layer_types):
            raise ValueError(f"layer_types has {len(layer_types)} entries, model has {len(base_layer_types)}")
        return layer_types

    if "full_attention_interval" in variant:
        interval = int(variant["full_attention_interval"])
        if interval == 0:
            return ["linear_attention"] * len(base_layer_types)
        # Match the stock convention: the layer at the end of each block is full attention.
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(len(base_layer_types))
        ]

    raise ValueError(f"unrecognised layer_variant: {variant}")


class Qwen35Evaluator(BaseEvaluator):
    """Local inference for the Qwen3.5 family (natively image-text-to-text)."""

    def __init__(self, model_id, max_image_size=512, max_new_tokens=100,
                 dtype=torch.bfloat16, layer_variant=None, name=None, device_map="cuda", thinking=None):
        super().__init__(model_id)
        self.max_image_size = max_image_size
        self.max_new_tokens = max_new_tokens
        # None (the default) keeps the template's empty think block, as every run before 2026-10-08 had.
        self.thinking = {**THINKING_DEFAULTS, **thinking} if thinking and thinking.get("enabled") else None
        self.dtype = dtype
        self.layer_variant = layer_variant
        self.name = name or model_id.split("/")[-1].lower()
        # "cuda" keeps the model on one GPU, which is what every model up to 9B needs. "auto" lets
        # accelerate shard across the visible GPUs, for weights that do not fit one card: the 27B is
        # ~56 GB, so it needs two 48 GB A6000s (the 96 GB R6000 nodes are usually unavailable).
        self.device_map = device_map
        self.converted_layers = 0
        self._load_model()

    def _load_model(self):
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        config = AutoConfig.from_pretrained(self.model_id)
        text_config = getattr(config, "text_config", config)

        if self.layer_variant:
            base = list(text_config.layer_types)
            new = resolve_layer_types(base, self.layer_variant)
            self.converted_layers = sum(1 for a, b in zip(base, new) if a != b)
            text_config.layer_types = new
            # A converted layer's weights do not exist in the checkpoint, so transformers
            # initialises them randomly. Without further training the variant is not a
            # trained model; it is only meaningful after finetuning (see README).
            print(f"WARNING: layer_variant converts {self.converted_layers}/{len(base)} layers; "
                  "their weights are randomly initialised, not pretrained")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, config=config, dtype=self.dtype, device_map=self.device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # The checkpoint's eos_token_id (248044) is not the token the chat template ends
        # turns with (<|im_end|>), so without this the model runs past its turn and starts
        # writing the next one. Stop on either.
        tokenizer = self.processor.tokenizer
        eos = self.model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        for token in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
            if token is not None and token not in eos:
                eos.append(token)
        self.eos_token_id = eos
        self.pad_token_id = tokenizer.pad_token_id
        self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")

        print(f"Loaded {self.model_id} on {next(self.model.parameters()).device} (eos={eos})")

    def _encode_image(self, image):
        if max(image.size) > self.max_image_size:
            image = image.copy()
            image.thumbnail((self.max_image_size,) * 2, Image.Resampling.LANCZOS)
        return {"type": "image", "image": image}

    def _encode_tokens(self, ids):
        """A stimulus given as token IDs (tokens/stimuli.py), spliced in as-is by _prepare_inputs."""
        return {"type": "tokens", "ids": list(ids)}

    def check_image_capacity(self, n_images, sample_image=None):
        """Probe with n_images copies of a stimulus, so the probe costs what the real trial costs.

        `sample_image` should be an image from the condition's own dataset. Without one the probe
        falls back to a blank at max_image_size, which over-states the cost of any smaller stimulus:
        a 512px blank is 258 tokens against 66 for a 256px Brady object, so probing 2,003 Brady
        images exceeded the context window and reported a condition as impossible that in fact
        needed 30 GB.
        """
        probe = sample_image or Image.new("RGB", (self.max_image_size,) * 2, (255, 255, 255))
        content = [{"type": "text", "text": "Reply with the number 1."}]
        content += [self._encode_image(probe) for _ in range(n_images)]
        try:
            self._call_api([{"role": "user", "content": content}])
            return True
        except Exception as e:
            if "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError":
                torch.cuda.empty_cache()
                return False
            raise

    def _collect_images(self, messages):
        return [
            item["image"]
            for msg in messages if isinstance(msg["content"], list)
            for item in msg["content"] if isinstance(item, dict) and item.get("type") == "image"
        ]

    def _prepare_inputs(self, messages, prefill=None, thought=None, opening_think=False):
        """Render the prompt. With thinking on, the template leaves a think block open; `thought`
        (from think()) closes it, so the answer then sits where it does with thinking off.
        `opening_think` is think() itself asking for the open block."""
        if self.thinking and thought is None and not opening_think:
            raise ValueError("thinking is enabled but no thought was passed; call think() first "
                             "(this task may not support thinking)")
        images = self._collect_images(messages)
        token_items = []
        messages = self._stub_token_items(messages, token_items)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(self.thinking),
        )
        if thought is not None:
            text += thought["prefill"]
        if prefill:
            text += prefill
        if token_items:
            if images:
                raise ValueError("a prompt mixing images and token items is not supported")
            return self._splice_token_items(text, token_items).to(self.model.device)
        kwargs = {"text": [text], "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs).to(self.model.device)

    def _stub_token_items(self, messages, token_items):
        """Copy of messages with each token item replaced by a numbered text sentinel."""
        def stub(item):
            if isinstance(item, dict) and item.get("type") == "tokens":
                token_items.append(item["ids"])
                return {"type": "text", "text": TOKENS_SENTINEL.format(len(token_items) - 1)}
            return item
        return [{**msg, "content": [stub(item) for item in msg["content"]]} if isinstance(msg["content"], list)
                else msg for msg in messages]

    def _splice_token_items(self, text, token_items):
        """Tokenize the text around the sentinels and put each item's IDs in its place, untouched.

        Tokenizing the decoded item instead would let its pieces merge with each other and with the
        surrounding text, changing the very IDs that define old vs new.
        """
        tokenizer = self.processor.tokenizer
        parts = TOKENS_SENTINEL_RE.split(text)
        ids = []
        for k, part in enumerate(parts):
            if k % 2:
                ids += token_items[int(part)]
            elif part:
                ids += tokenizer.encode(part, add_special_tokens=False)
        input_ids = torch.tensor([ids])
        # The same three fields the processor returns for text-only input.
        return BatchFeature({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                             "mm_token_type_ids": torch.zeros_like(input_ids)})

    def _choice_token_ids(self, choice):
        """Token ids that stand for this choice alone.

        `encode(" 1")` is [space, "1"] in this tokenizer, so taking the first token of the spaced
        form put the shared space token in every choice's set, where its ~0.997 probability drowned
        the digits and made every choice look like a 50/50 call.
        """
        tokenizer = self.processor.tokenizer
        candidates = {tokenizer.encode(choice, add_special_tokens=False)[0],
                      tokenizer.encode(f" {choice}", add_special_tokens=False)[-1]}
        ids = sorted(i for i in candidates if tokenizer.decode([i]).strip() == choice)
        if not ids:
            raise ValueError(f"no token decodes to choice {choice!r}")
        return ids

    def think(self, messages, trial_seed):
        """Generate a thinking trace for this prompt; None when thinking is off.

        Sampled with torch seeded from `trial_seed` (e.g. "0:17" for seed 0, trial 17), so a trial's
        trace reproduces without the run consuming the task's own RNG, which would change the episodes.
        A trace that reaches the budget is closed with </think> and the trial is still scored;
        `stop` records which happened ("closed", "budget", or "eos" if the turn ended inside the block).
        """
        if not self.thinking:
            return None
        cfg = self.thinking
        inputs = self._prepare_inputs(messages, opening_think=True)
        torch.manual_seed(int.from_bytes(hashlib.sha256(str(trial_seed).encode()).digest()[:8], "big"))
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=cfg["budget"], do_sample=True,
                temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                eos_token_id=[self.think_end_id, *self.eos_token_id], pad_token_id=self.pad_token_id,
            )
        ids = out[0, inputs.input_ids.shape[1]:].tolist()
        if self.think_end_id in ids:
            stop = "closed"
            ids = ids[:ids.index(self.think_end_id)]
        elif ids and ids[-1] in self.eos_token_id:
            stop = "eos"
            ids = ids[:-1]
        else:
            stop = "budget"
        trace = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        close = "</think>\n\n" if trace.endswith("\n") else "\n</think>\n\n"
        return {"trace": trace, "n_tokens": len(ids), "stop": stop, "seed": str(trial_seed),
                "prefill": trace + close}

    def score_choices(self, messages, choices=("1", "2"), prefill=CHOICE_PREFILL, thought=None):
        """Forced choice: probability of each choice as the first token after `prefill`.

        Free generation lets the model answer with a paragraph that then has to be parsed, which
        silently turned format failures into memory scores. This reads the answer off the logits
        instead, so every trial yields a choice and nothing needs parsing.
        """
        inputs = self._prepare_inputs(messages, prefill=prefill, thought=thought)
        with torch.no_grad():
            # Only the last position's logits are needed. Without this the model materialises
            # logits for every position: at 250 THINGS images that is 64k tokens x 248k vocab,
            # ~30 GiB, which OOMs a 48 GiB GPU. generate() keeps one position for the same reason.
            out = self.model(**inputs, logits_to_keep=1)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        # Summed over both tokenisations, then renormalised over the choices, so the number
        # reported is the model's preference between them, not its absolute confidence.
        raw = {c: float(probs[self._choice_token_ids(c)].sum()) for c in choices}
        total = sum(raw.values())
        norm = {c: (v / total if total > 0 else 1 / len(choices)) for c, v in raw.items()}
        chosen = max(norm, key=norm.get)
        # choice_mass is how much of the model's next-token mass the choices hold at all. When it
        # is low the choice is being read off the tail of the distribution and is unreliable;
        # top_tokens shows where the rest went (e.g. the 2B at n=5 put up to 64% elsewhere).
        top = torch.topk(probs, 5)
        tokenizer = self.processor.tokenizer
        top_tokens = [[tokenizer.decode([int(i)]), round(float(p), 4)] for p, i in zip(top.values, top.indices)]
        return {"choice": chosen, "probs": norm, "choice_mass": total, "top_tokens": top_tokens, "prefill": prefill}

    def _call_api(self, messages, thought=None):
        inputs = self._prepare_inputs(messages, thought=thought)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
            )
        reply = self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )[0]
        return reply.strip()

    def get_name(self):
        return self.name


def build_evaluator(model_cfg):
    return Qwen35Evaluator(
        model_id=model_cfg["model_id"],
        max_image_size=model_cfg.get("max_image_size", 512),
        max_new_tokens=model_cfg.get("max_new_tokens", 100),
        layer_variant=model_cfg.get("layer_variant"),
        name=model_cfg["name"],
        device_map=model_cfg.get("device_map", "cuda"),
        thinking=model_cfg.get("thinking"),
    )
