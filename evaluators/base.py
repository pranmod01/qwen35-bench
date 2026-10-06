from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image

# Load .env file from project root
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on environment variables


class BaseEvaluator(ABC):
    """Base class for VLM evaluators."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def _encode_image(self, image: Image.Image) -> Any:
        """Encode PIL Image to API-specific format."""
        pass

    @abstractmethod
    def _call_api(self, messages: List[Dict]) -> str:
        """Make API call and return response text."""
        pass

    def get_name(self) -> str:
        return self.model_id

    def check_image_capacity(self, n_images: int) -> bool:
        """Probe whether the model can handle n_images in a single request.

        Returns True on success, False only when the provider explicitly rejects
        the request as too large (400/413 with capacity-related wording).
        Other errors (rate limits, 5xx, network) propagate so callers can react.
        """
        pixel = Image.new("RGB", (1, 1), (255, 255, 255))
        encoded = [self._encode_image(pixel) for _ in range(n_images)]
        content = [{"type": "text", "text": "Reply with the number 1."}] + encoded
        messages = [{"role": "user", "content": content}]
        try:
            self._call_api(messages)
            return True
        except Exception as e:
            status = getattr(e, "status_code", None)
            msg = str(e).lower()
            capacity_phrases = ("too many images", "too large", "payload too large",
                                "context length", "context window", "maximum context",
                                "exceeds the maximum", "request too large",
                                "too much media", "exceed max allowed size",
                                "exceed the max allowed size")
            if status in (400, 413) and any(p in msg for p in capacity_phrases):
                return False
            raise
