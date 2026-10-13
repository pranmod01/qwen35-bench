import os
import json
import random
import re
from numbers import Number
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load .env for HF_TOKEN
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

class DirectoryDataset():
    def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in extensions
        ])
        if not self.image_paths:
            print(f"Warning: No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_metadata(self, index):
        return {"path": str(self.image_paths[index]), "name": self.image_paths[index].name}


class ImageNetConceptsDataset:
    """One 256 px center crop per mutually distinct ImageNet-21k concept, as listed in concepts.json.

    Built by scripts/build_imagenet_stimuli.py and selected by scripts/select_imagenet_concepts.py; the
    directory holds crops for every class, but only the selected concepts are items here. One image per
    concept, so only novel foils exist.
    """
    def __init__(self, root='memory_datasets/ImageNet21k_w21_256'):
        self.root = Path(root)
        self.concepts = json.loads((self.root / "concepts.json").read_text())["concepts"]

    def __len__(self):
        return len(self.concepts)

    def get_image(self, index, exemplar_index=0):
        return Image.open(self.root / f"{self.concepts[index]['wnid']}.jpg").convert("RGB")

    def get_metadata(self, index, exemplar_index=0):
        c = self.concepts[index]
        return {"wnid": c["wnid"], "category": c["lemma"].split(",")[0], "domain": c["domain"], "category_id": index}


class ThingsDataset:
    """Gets images from a local THINGS install when available, else via Hugging Face."""
    def __init__(
        self,
        n_categories=None,
        exemplars_per_category=1,
        local_root='memory_datasets/THINGS/object_images',
        source="auto",
        excluded_categories=None,
    ):
        self.local_root = Path(local_root)
        self.source = source
        self.excluded_categories = {self._normalize_category_name(category) for category in (excluded_categories or [])}
        self.category_metadata = {}
        try:
            if source not in {"auto", "local", "streaming"}:
                raise ValueError(f"Unsupported THINGS source={source!r}. Expected 'auto', 'local', or 'streaming'.")

            if source in {"auto", "local"} and self._load_from_local_cache(
                n_categories=n_categories,
                exemplars_per_category=exemplars_per_category,
            ):
                print(f"Loaded THINGS from local files: {self.local_root}")
                return
            if source == "auto":
                print(f"Local THINGS files not available at {self.local_root}; falling back to Hugging Face streaming.")
            if source == "local":
                raise ValueError(f"Could not load THINGS from local files: {self.local_root}")

            from datasets import load_dataset
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            # Prioritize the one the user mentioned
            names = ["Haitao999/things-eeg", "He-Bart/THINGS", "Hebart/THINGS", "THINGS-data/THINGS", "RAIlab/THINGS"]
            self.ds = None
            self.name = None
            id_to_name = None
            for name in names:
                try:
                    # Use streaming=True to avoid downloading 30GB at once
                    self.ds = load_dataset(name, split="train", streaming=True, token=hf_token)
                    self.name = name
                    if hasattr(self.ds, 'features') and 'label' in self.ds.features:
                        label_feature = self.ds.features['label']
                        if hasattr(label_feature, 'names'):
                            id_to_name = label_feature.names
                    print(f"Successfully connected to THINGS via {name}")
                    break
                except:
                    continue
            
            if self.ds is None:
                raise ValueError("Could not find THINGS dataset on Hugging Face. Please verify the dataset name.")
            
            # Map category name to list of images
            self.category_groups = {} 
            self.category_names = [] # Maintain order of arrival
            self.category_metadata = {}

            print("Fetching images from streaming dataset...")
            for item in self.ds:
                # Determine category name
                label = item.get('label')
                if label is not None and id_to_name:
                    cat = id_to_name[label]
                else:
                    cat = item.get('category') or item.get('concept') or f"cat_{label}"
                cat = self._normalize_category_name(cat)

                if cat in self.excluded_categories:
                    continue
                
                if cat not in self.category_groups:
                    if n_categories and len(self.category_groups) >= n_categories:
                        continue
                    self.category_groups[cat] = []
                    self.category_metadata[cat] = []
                    self.category_names.append(cat)
                
                if len(self.category_groups[cat]) < exemplars_per_category:
                    img = item['image']
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.array(img))
                    self.category_groups[cat].append(img.convert("RGB"))
                    self.category_metadata[cat].append(self._build_stream_metadata(item=item, category=cat))
                
                # Check if we have enough
                if n_categories and len(self.category_groups) >= n_categories:
                    # Check if all have enough exemplars
                    if all(len(self.category_groups[c]) >= exemplars_per_category for c in self.category_names):
                        break
            
            print(f"Loaded {len(self.category_groups)} categories with up to {exemplars_per_category} exemplars each.")
            
        except Exception as e:
            print(f"Error loading THINGS dataset: {e}")
            self.category_groups = {}
            self.category_names = []
            self.category_metadata = {}

    def _normalize_category_name(self, category):
        return re.sub(r"^\d+_", "", str(category))

    def _jsonable_scalar(self, value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Number):
            return value.item() if hasattr(value, "item") else float(value)
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def _build_stream_metadata(self, item, category):
        metadata = {
            "source": "streaming",
            "dataset_name": self.name,
            "category": category,
        }
        for key, value in item.items():
            if key == "image":
                continue
            if isinstance(value, (list, dict, tuple)):
                continue
            metadata[key] = self._jsonable_scalar(value)

        image_obj = item.get("image")
        for attr_name, key_name in (
            ("filename", "image_filename"),
            ("path", "image_path"),
            ("format", "image_format"),
        ):
            attr_value = getattr(image_obj, attr_name, None)
            if attr_value is not None:
                metadata[key_name] = self._jsonable_scalar(attr_value)
        return metadata

    def _load_from_local_cache(self, n_categories=None, exemplars_per_category=1):
        if not self.local_root.exists():
            return False

        category_dirs = sorted([path for path in self.local_root.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
        if not category_dirs:
            return False
        # Taking the first n in sorted order made every small-n episode reuse the same concepts
        # (n=1 was always 'aardvark'). Shuffle via the global RNG so the eval seed controls it.
        if n_categories:
            random.shuffle(category_dirs)

        self.category_groups = {}
        self.category_names = []
        self.category_metadata = {}
        for category_dir in category_dirs:
            image_paths = sorted(
                [
                    path
                    for path in category_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ],
                key=lambda p: p.name.lower(),
            )
            if len(image_paths) < exemplars_per_category:
                continue
            if n_categories and len(self.category_groups) >= n_categories:
                break

            category_name = self._normalize_category_name(category_dir.name)
            if category_name in self.excluded_categories:
                continue
            self.category_names.append(category_name)
            chosen_paths = image_paths[:exemplars_per_category]
            self.category_groups[category_name] = [
                Image.open(path).convert("RGB") for path in chosen_paths
            ]
            self.category_metadata[category_name] = [
                {
                    "source": "local_files",
                    "category": category_name,
                    "local_path": str(path),
                    "local_filename": path.name,
                }
                for path in chosen_paths
            ]

        return bool(self.category_groups)

    def __len__(self):
        return len(self.category_names)

    def get_image(self, index, exemplar_index=0):
        cat = self.category_names[index]
        exemplars = self.category_groups[cat]
        if isinstance(exemplars[0], Path):
            return Image.open(exemplars[exemplar_index % len(exemplars)]).convert("RGB")
        return exemplars[exemplar_index % len(exemplars)]

    def get_metadata(self, index, exemplar_index=0):
        cat = self.category_names[index]
        exemplars = self.category_groups[cat]
        exemplar_id = exemplar_index % len(exemplars)
        metadata = {"category": cat, "category_id": index, "exemplar_id": exemplar_id}
        extra = self.category_metadata.get(cat, [])
        if exemplar_id < len(extra):
            metadata.update(extra[exemplar_id])
        return metadata


class CuedRecallImageNetDataset:
    """Streams maxbennett/cued-recall-imagenet, organized by class.

    The HF dataset is a WebDataset of ImageNet shards: each row has a JPG
    image and a __key__ like "class_0000/img_00000165". We stream lazily and
    cache decoded PIL images in memory for the classes we sample.
    """
    REPO_ID = "maxbennett/cued-recall-imagenet"

    def __init__(self, n_categories=None, exemplars_per_category=1, split="train"):
        self.n_categories = n_categories
        self.exemplars_per_category = exemplars_per_category
        self.split = split
        self.category_groups = {}
        self.category_names = []
        self.category_metadata = {}
        self._load()

    def _load(self):
        from datasets import load_dataset
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        ds = load_dataset(self.REPO_ID, split=self.split, streaming=True, token=hf_token)

        print(f"Streaming {self.REPO_ID} (split={self.split}, "
              f"n_categories={self.n_categories}, exemplars_per_category={self.exemplars_per_category})...")

        for item in ds:
            key = item.get("__key__") or ""
            if "/" not in key:
                continue
            cat = key.split("/", 1)[0]

            if cat not in self.category_groups:
                if self.n_categories and len(self.category_groups) >= self.n_categories:
                    continue
                self.category_groups[cat] = []
                self.category_metadata[cat] = []
                self.category_names.append(cat)

            if len(self.category_groups[cat]) < self.exemplars_per_category:
                img = item.get("jpg")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                self.category_groups[cat].append(img.convert("RGB"))
                self.category_metadata[cat].append({
                    "source": "streaming",
                    "dataset_name": self.REPO_ID,
                    "category": cat,
                    "key": key,
                })

            if (self.n_categories
                    and len(self.category_groups) >= self.n_categories
                    and all(len(self.category_groups[c]) >= self.exemplars_per_category
                            for c in self.category_names)):
                break

        print(f"Loaded {len(self.category_groups)} cued-recall-imagenet classes "
              f"with up to {self.exemplars_per_category} exemplars each.")

    def __len__(self):
        return len(self.category_names)

    def get_image(self, index, exemplar_index=0):
        cat = self.category_names[index]
        exemplars = self.category_groups[cat]
        return exemplars[exemplar_index % len(exemplars)]

    def get_metadata(self, index, exemplar_index=0):
        cat = self.category_names[index]
        exemplars = self.category_groups[cat]
        exemplar_id = exemplar_index % len(exemplars)
        metadata = {"category": cat, "category_id": index, "exemplar_id": exemplar_id}
        extra = self.category_metadata.get(cat, [])
        if exemplar_id < len(extra):
            metadata.update(extra[exemplar_id])
        return metadata


class BradyDataset:
    """Handles Brady2008 and Brady2013 datasets."""
    HF_COLLECTIONS = {
        "Objects": "brady_objects",
        "Exemplar": "brady_exemplar",
        "State": "brady_state",
        "Brady2013ColorObjects": "brady_color_objects",
    }

    def __init__(self, type='Objects', root_dir='memory_datasets', source="local",
                 repo_id="chrisiyer/vision-memory-tasks"):
        self.source = source
        self.repo_id = repo_id
        if source not in {"local", "hf"}:
            raise ValueError(f"Unsupported BradyDataset source={source!r}. Expected 'local' or 'hf'.")
        if source == "hf":
            self._load_from_hf(type)
            return


        self.root = Path(root_dir)
        if 'Brady' in type:
             self.path = self.root / type
        else:
             # Default to Brady2008 prefix if just 'Objects', 'Exemplar', 'State'
             self.path = self.root / f"Brady2008{type}"

        self.image_paths = sorted([p for p in self.path.glob('*') if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
        self.pair_paths = self._build_pair_paths()

    def __len__(self):
        return len(self.records) if self.source == "hf" else len(self.image_paths)

    def get_image(self, index):
        if self.source == "hf":
            return Image.open(self._hf_image_path(self.records[index]["file_name"])).convert("RGB")
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_metadata(self, index):
        if self.source == "hf":
            record = self.records[index]
            return {
                "source": "hf",
                "dataset_repo": self.repo_id,
                "image_id": record["image_id"],
                "name": record["original_file_name"],
                "path": self._hf_image_path(record["file_name"]),
            }
        return {"path": str(self.image_paths[index]), "name": self.image_paths[index].name}

    def _build_pair_paths(self):
        if self.path.name not in {"Brady2008Exemplar", "Brady2008State"}:
            return []

        groups = {}
        for path in self.image_paths:
            match = re.match(r"^(.*?)(\d+)?$", path.stem)
            base_name = match.group(1).lower() if match else path.stem.lower()
            groups.setdefault(base_name, []).append(path)

        pair_paths = []
        for base_name in sorted(groups):
            paths = sorted(groups[base_name], key=lambda p: p.name.lower())
            if len(paths) == 2:
                pair_paths.append(tuple(paths))
        return pair_paths

    def get_pair(self, index):
        if self.source == "hf":
            if not self.pair_records:
                raise ValueError("Pair access is only supported for Brady2008Exemplar and Brady2008State.")
            image_ids = self.pair_records[index]["image_ids"]
            return self.get_image(self.image_id_to_index[image_ids[0]]), self.get_image(self.image_id_to_index[image_ids[1]])

        if not self.pair_paths:
            raise ValueError("Pair access is only supported for Brady2008Exemplar and Brady2008State.")
        original_path, foil_path = self.pair_paths[index]
        return Image.open(original_path).convert("RGB"), Image.open(foil_path).convert("RGB")

    def _hf_collection_id(self, type):
        if type in self.HF_COLLECTIONS:
            return self.HF_COLLECTIONS[type]
        if type.startswith("Brady2008"):
            suffix = type.replace("Brady2008", "")
            if suffix in self.HF_COLLECTIONS:
                return self.HF_COLLECTIONS[suffix]
        raise ValueError(f"Unsupported Brady HF dataset type={type!r}.")

    def _load_from_hf(self, type):
        from huggingface_hub import hf_hub_download

        self.hf_hub_download = hf_hub_download
        self.collection_id = self._hf_collection_id(type)
        metadata_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=f"data/{self.collection_id}/metadata.jsonl",
        )
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        self.image_id_to_index = {record["image_id"]: index for index, record in enumerate(self.records)}

        self.pair_records = []
        if self.collection_id in {"brady_exemplar", "brady_state"}:
            pairs_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=f"data/{self.collection_id}/pairs.jsonl",
            )
            with open(pairs_path, "r", encoding="utf-8") as f:
                self.pair_records = [json.loads(line) for line in f if line.strip()]
        self.pair_paths = self.pair_records

    def _hf_image_path(self, file_name):
        return self.hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=f"data/{self.collection_id}/{file_name}",
        )


class LaMemDataset():
    """
    Skeleton for LaMem dataset.
    In a real scenario, this would handle downloading and metadata parsing.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.metadata_file = self.root_dir / "lamem_score.txt" # Common filename for LaMem

        self.image_paths = []
        self.scores = {} # image_name -> score

        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                for line in f:
                    # Format: folder/image_name score
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_rel_path, score = parts
                        img_path = self.image_dir / img_rel_path
                        if img_path.exists():
                            self.image_paths.append(img_path)
                            self.scores[img_path.name] = float(score)

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_metadata(self, index):
        path = self.image_paths[index]
        return {
            "path": str(path),
            "name": path.name,
            "memorability": self.scores.get(path.name)
        }

    def get_ground_truth(self):
        """Returns a dict of {image_name: score} for all images in the dataset."""
        return self.scores


def generate_color_palette(n_colors=36):
    """Generates a visual palette with numbered colors."""
    import colorsys

    tile_size = 100
    cols = 6
    rows = (n_colors + cols - 1) // cols
    palette = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(palette)

    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    hues = np.linspace(0, 1.0, n_colors, endpoint=False)

    for i, hue in enumerate(hues):
        r_idx = i // cols
        c_idx = i % cols

        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = tuple(int(x * 255) for x in rgb)

        x0, y0 = c_idx * tile_size, r_idx * tile_size
        x1, y1 = x0 + tile_size, y0 + tile_size
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
        draw.text((x0 + 5, y0 + 5), str(i + 1), fill=(0, 0, 0), font=font)

    return palette


# Helper to create a dummy dataset for testing
# (Removed as requested)
