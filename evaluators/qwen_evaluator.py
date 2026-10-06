import torch
from typing import List, Dict, Any
from PIL import Image
from .base import BaseEvaluator


class QwenEvaluator(BaseEvaluator):
    """Qwen3-VL local inference evaluator.

    Requires: transformers, torch, qwen_vl_utils
    GPU recommended for reasonable inference speed.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
                 max_image_size: int = 512, vision_chunk_size=None,
                 attn_implementation=None):
        super().__init__(model_id)
        self.model = None
        self.processor = None
        self.max_image_size = max_image_size
        self.vision_chunk_size = vision_chunk_size
        self.attn_implementation = attn_implementation
        self._load_model()
        if vision_chunk_size:
            self._install_chunked_vision_encoder(vision_chunk_size)

    def _load_model(self):
        """Load Qwen3-VL model and processor."""
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"Loading {self.model_id}...")

        kwargs = dict(dtype=torch.bfloat16, device_map="auto")
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"Model loaded on {next(self.model.parameters()).device}")

    def _install_chunked_vision_encoder(self, chunk_size: int):
        """Monkey-patch get_image_features so the vision tower processes images
        in fixed-size chunks. Avoids OOM in the vision encoder when many images
        are passed at once; results are mathematically identical because each
        image's patches are processed independently inside the encoder.
        """
        inner = self.model.model  # Qwen3VLModel
        visual = inner.visual
        merge_sq = visual.spatial_merge_size ** 2

        def chunked_get_image_features(pixel_values, image_grid_thw=None):
            if image_grid_thw is None or len(image_grid_thw) <= chunk_size:
                pixel_values_typed = pixel_values.type(visual.dtype)
                embeds, deepstack = visual(pixel_values_typed, grid_thw=image_grid_thw)
                split_sizes = (image_grid_thw.prod(-1) // merge_sq).tolist()
                return torch.split(embeds, split_sizes), deepstack

            patches_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] *
                                 image_grid_thw[:, 2]).tolist()

            all_embed_splits = []
            all_deepstack_chunks = None
            pixel_idx = 0
            for chunk_start in range(0, len(image_grid_thw), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(image_grid_thw))
                chunk_grid = image_grid_thw[chunk_start:chunk_end]
                chunk_patch_count = sum(patches_per_image[chunk_start:chunk_end])
                chunk_pixels = pixel_values[pixel_idx:pixel_idx + chunk_patch_count]
                pixel_idx += chunk_patch_count

                embeds, deepstack = visual(chunk_pixels.type(visual.dtype),
                                           grid_thw=chunk_grid)
                split_sizes = (chunk_grid.prod(-1) // merge_sq).tolist()
                all_embed_splits.extend(torch.split(embeds, split_sizes))

                if all_deepstack_chunks is None:
                    all_deepstack_chunks = [[d] for d in deepstack]
                else:
                    for i, d in enumerate(deepstack):
                        all_deepstack_chunks[i].append(d)
                del embeds, deepstack

            deepstack_concat = [torch.cat(parts, dim=0) for parts in all_deepstack_chunks]
            return tuple(all_embed_splits), deepstack_concat

        inner.get_image_features = chunked_get_image_features
        print(f"Chunked vision encoder enabled (chunk_size={chunk_size})")

    def _encode_image(self, image: Image.Image) -> Dict[str, Any]:
        """Return image in Qwen-compatible format.

        Qwen expects images as PIL Images in the message content.
        """
        max_size = self.max_image_size
        if max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return {"type": "image", "image": image}

    def check_image_capacity(self, n_images: int) -> bool:
        """Probe with realistic images so GPU OOM is caught accurately.

        The base-class probe uses 1x1 px which produces far fewer image tokens
        than real usage; that lets the probe pass at sequence lengths that then
        OOM during the actual evaluation. We use the evaluator's configured
        max_image_size so the probe matches the inference path.
        """
        s = self.max_image_size
        test_img = Image.new("RGB", (s, s), (255, 255, 255))
        encoded = [self._encode_image(test_img) for _ in range(n_images)]
        content = [{"type": "text", "text": "Reply with the number 1."}] + encoded
        messages = [{"role": "user", "content": content}]
        try:
            self._call_api(messages)
            return True
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg or e.__class__.__name__ == "OutOfMemoryError":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return False
            raise

    def _call_api(self, messages: List[Dict]) -> str:
        """Run local inference and return response text."""
        # Convert messages to Qwen format
        qwen_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                qwen_messages.append({"role": role, "content": content})
            else:
                # Build content list with text and images
                qwen_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            qwen_content.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image":
                            qwen_content.append({"type": "image", "image": item["image"]})
                        elif item.get("type") == "image_url":
                            # Handle base64 encoded images (convert back to PIL)
                            import base64
                            from io import BytesIO
                            url = item["image_url"]["url"]
                            if url.startswith("data:"):
                                b64_data = url.split(",")[1]
                                img_bytes = base64.b64decode(b64_data)
                                img = Image.open(BytesIO(img_bytes))
                                qwen_content.append({"type": "image", "image": img})
                    else:
                        qwen_content.append(item)

                qwen_messages.append({"role": role, "content": qwen_content})

        # Process with Qwen processor
        text_prompt = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Collect all images from messages
        images = []
        for msg in qwen_messages:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item["image"])

        # Process inputs
        if images:
            inputs = self.processor(
                text=[text_prompt],
                images=images,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_prompt],
                padding=True,
                return_tensors="pt"
            )

        inputs = inputs.to(self.model.device)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        # Decode only the new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response.strip()

    def get_name(self) -> str:
        """Return short name for results."""
        model_upper = self.model_id.upper()
        if "8B" in model_upper:
            return "qwen3-vl-8b"
        elif "4B" in model_upper:
            return "qwen3-vl-4b"
        return "qwen3-vl"
