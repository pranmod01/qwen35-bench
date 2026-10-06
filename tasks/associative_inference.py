import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import BradyDataset, DirectoryDataset, ThingsDataset


class AssociativeInferenceTask:
    def __init__(self, dataset_name="Brady2008", n_trials=20, pair_type="image",
                 wordpool_path="memory_datasets/wasnorm_wordpool.txt", image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks"):
        if n_trials % 2 != 0:
            raise ValueError("n_trials must be even so the study phase can split into AB and BC pairs.")

        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.n_pairs = n_trials // 2
        self.pair_type = pair_type
        self.image_dir = image_dir
        self.source = source
        self.repo_id = repo_id
        self.dataset = self._load_dataset()

        if pair_type == "word":
            self.wordpool = self._load_wordpool(wordpool_path)

    def _load_dataset(self):
        n_required = self.n_pairs * 2 if self.pair_type == "word" else self.n_pairs * 3
        if self.image_dir:
            return DirectoryDataset(self.image_dir)
        if self.dataset_name == "Brady2008":
            return BradyDataset(type="Objects", source=self.source, repo_id=self.repo_id)
        return ThingsDataset(n_categories=n_required)

    def _load_wordpool(self, wordpool_path):
        if self.source == "hf":
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="data/wordpool/wasnorm_wordpool.txt",
            )
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]

        wordpool_path = Path(wordpool_path)
        if wordpool_path.exists():
            with open(wordpool_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        print(f"Warning: Wordpool not found at {wordpool_path}, using dummy words.")
        return [f"WORD{i}" for i in range(1000)]

    def get_trials(self):
        if self.pair_type == "word":
            return self._get_word_trials()
        return self._get_image_trials()

    def _get_image_trials(self):
        n_available = len(self.dataset)
        n_pairs = min(self.n_pairs, n_available // 3)
        if n_pairs < 1:
            raise ValueError("Associative inference requires at least 1 latent ABC chain.")

        indices = list(range(n_available))
        random.shuffle(indices)
        selected_indices = indices[:n_pairs * 3]

        a_indices = selected_indices[:n_pairs]
        b_indices = selected_indices[n_pairs:2 * n_pairs]
        c_indices = selected_indices[2 * n_pairs:3 * n_pairs]

        chain_items = []
        for chain_index in range(n_pairs):
            chain_items.append(
                {
                    "chain_index": chain_index,
                    "A": {
                        "image": self.dataset.get_image(a_indices[chain_index]),
                        "metadata": {
                            **self.dataset.get_metadata(a_indices[chain_index]),
                            "role": "A",
                            "chain_index": chain_index,
                        },
                    },
                    "B": {
                        "image": self.dataset.get_image(b_indices[chain_index]),
                        "metadata": {
                            **self.dataset.get_metadata(b_indices[chain_index]),
                            "role": "B",
                            "chain_index": chain_index,
                        },
                    },
                    "C": {
                        "image": self.dataset.get_image(c_indices[chain_index]),
                        "metadata": {
                            **self.dataset.get_metadata(c_indices[chain_index]),
                            "role": "C",
                            "chain_index": chain_index,
                        },
                    },
                }
            )

        study_sequence = []
        for chain in chain_items:
            study_sequence.append(
                {
                    "images": [chain["A"]["image"], chain["B"]["image"]],
                    "pair_type": "AB",
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "left_role": "A",
                        "right_role": "B",
                        "left_item": chain["A"]["metadata"],
                        "right_item": chain["B"]["metadata"],
                    },
                }
            )
        for chain in chain_items:
            study_sequence.append(
                {
                    "images": [chain["B"]["image"], chain["C"]["image"]],
                    "pair_type": "BC",
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "left_role": "B",
                        "right_role": "C",
                        "left_item": chain["B"]["metadata"],
                        "right_item": chain["C"]["metadata"],
                    },
                }
            )

        fallback_foil = None
        if n_pairs == 1:
            unused_indices = indices[n_pairs * 3:]
            if not unused_indices:
                raise ValueError("Associative inference with one image chain requires one extra foil image.")
            foil_idx = unused_indices[0]
            fallback_foil = {
                "image": self.dataset.get_image(foil_idx),
                "metadata": {
                    **self.dataset.get_metadata(foil_idx),
                    "role": "foil_C",
                    "chain_index": None,
                },
            }

        test_phase = self._build_test_phase(chain_items, key="C", image_key="image", fallback_foil=fallback_foil)

        return {
            "study_prompt": (
                "Remember these pairs, and pay attention to hidden connections between them. First you will see a sequence of image-image pairs. "
                "Then you will be tested on your memory for connections between items."
            ),
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }

    def _get_word_trials(self):
        n_available = len(self.dataset)
        n_pairs = min(self.n_pairs, n_available // 2)
        if n_pairs < 1:
            raise ValueError("Associative inference requires at least 1 latent ABC chain.")

        indices = list(range(n_available))
        random.shuffle(indices)
        selected_indices = indices[:n_pairs * 2]

        a_indices = selected_indices[:n_pairs]
        b_indices = selected_indices[n_pairs:2 * n_pairs]
        words = random.sample(self.wordpool, n_pairs + (1 if n_pairs == 1 else 0))

        chain_items = []
        for chain_index in range(n_pairs):
            chain_items.append(
                {
                    "chain_index": chain_index,
                    "A": {
                        "image": self.dataset.get_image(a_indices[chain_index]),
                        "metadata": {
                            **self.dataset.get_metadata(a_indices[chain_index]),
                            "role": "A",
                            "chain_index": chain_index,
                        },
                    },
                    "B": {
                        "image": self.dataset.get_image(b_indices[chain_index]),
                        "metadata": {
                            **self.dataset.get_metadata(b_indices[chain_index]),
                            "role": "B",
                            "chain_index": chain_index,
                        },
                    },
                    "C": {
                        "word": words[chain_index],
                        "metadata": {"role": "C", "chain_index": chain_index},
                    },
                }
            )

        study_sequence = []
        for chain in chain_items:
            study_sequence.append(
                {
                    "images": [chain["A"]["image"], chain["B"]["image"]],
                    "pair_type": "AB",
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "left_role": "A",
                        "right_role": "B",
                        "left_item": chain["A"]["metadata"],
                        "right_item": chain["B"]["metadata"],
                    },
                }
            )
        for chain in chain_items:
            study_sequence.append(
                {
                    "image": chain["B"]["image"],
                    "word": chain["C"]["word"],
                    "pair_type": "BC",
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "left_role": "B",
                        "right_role": "C",
                        "left_item": chain["B"]["metadata"],
                        "right_item": chain["C"]["metadata"],
                    },
                }
            )

        fallback_word = None
        if n_pairs == 1:
            fallback_word = {
                "word": words[-1],
                "metadata": {"role": "foil_C", "chain_index": None},
            }

        test_phase = self._build_word_test_phase(chain_items, fallback_word=fallback_word)

        return {
            "study_prompt": (
                "Remember these pairs, and pay attention to hidden connections between them. First you will see a sequence of image-image or image-word pairs. "
                "Then you will be tested on your memory for connections between items."
            ),
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }

    def _build_test_phase(self, chain_items, key, image_key, fallback_foil=None):
        n_pairs = len(chain_items)
        test_phase = []

        for chain in chain_items:
            foil_options = [idx for idx in range(n_pairs) if idx != chain["chain_index"]]
            foil_chain = chain_items[random.choice(foil_options)] if foil_options else None

            correct_img = chain[key][image_key]
            foil_item = foil_chain[key] if foil_chain is not None else fallback_foil
            if foil_item is None:
                raise ValueError("Associative inference requires a foil option.")
            foil_img = foil_item[image_key]

            if random.random() < 0.5:
                images = [correct_img, foil_img]
                target = 1
            else:
                images = [foil_img, correct_img]
                target = 2

            test_phase.append(
                {
                    "cue_image": chain["A"]["image"],
                    "images": images,
                    "prompt": "Which of these two images is indirectly associated with the cue image? Reply with only the digit 1 or 2 and nothing else.",
                    "target": target,
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "cue_item": chain["A"]["metadata"],
                        "correct_option": chain[key]["metadata"],
                        "foil_option": foil_item["metadata"],
                    },
                }
            )

        return test_phase

    def _build_word_test_phase(self, chain_items, fallback_word=None):
        n_pairs = len(chain_items)
        test_phase = []

        for chain in chain_items:
            foil_options = [idx for idx in range(n_pairs) if idx != chain["chain_index"]]
            foil_chain = chain_items[random.choice(foil_options)] if foil_options else None

            correct_word = chain["C"]["word"]
            foil_item = foil_chain["C"] if foil_chain is not None else fallback_word
            if foil_item is None:
                raise ValueError("Associative inference requires a foil option.")
            foil_word = foil_item["word"]

            if random.random() < 0.5:
                options = [correct_word, foil_word]
                target = 1
            else:
                options = [foil_word, correct_word]
                target = 2

            test_phase.append(
                {
                    "cue_image": chain["A"]["image"],
                    "options": options,
                    "prompt": "Which of these two words is indirectly associated with the cue image? Reply with only the digit 1 or 2 and nothing else.",
                    "target": target,
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "cue_item": chain["A"]["metadata"],
                        "correct_option": chain["C"]["metadata"],
                        "foil_option": foil_item["metadata"],
                    },
                }
            )

        return test_phase


if __name__ == "__main__":
    task = AssociativeInferenceTask(n_trials=6)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"Test trials: {len(results['test_phase'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
