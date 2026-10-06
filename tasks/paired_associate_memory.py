import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import BradyDataset, DirectoryDataset, ThingsDataset


class PairedAssociateMemoryTask:
    def __init__(self, dataset_name='Brady2008', n_images=20, pair_type='word',
                 wordpool_path='memory_datasets/wasnorm_wordpool.txt', image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks"):
        self.n_images = n_images
        self.dataset_name = dataset_name
        self.pair_type = pair_type
        self.image_dir = image_dir
        self.source = source
        self.repo_id = repo_id

        n_to_load = n_images * 2 if pair_type == 'image' else n_images
        self.dataset = self._load_dataset(n_to_load)

        if pair_type == 'word':
            self.wordpool = self._load_wordpool(wordpool_path)

    def _load_dataset(self, n_to_load):
        if self.image_dir:
            return DirectoryDataset(self.image_dir)
        if self.dataset_name == 'Brady2008':
            return BradyDataset(type='Objects', source=self.source, repo_id=self.repo_id)
        return ThingsDataset(n_categories=n_to_load)

    def _load_wordpool(self, wordpool_path):
        if self.source == "hf":
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="data/wordpool/wasnorm_wordpool.txt",
            )
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]

        wordpool_path = Path(wordpool_path)
        if wordpool_path.exists():
            with open(wordpool_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        print(f"Warning: Wordpool not found at {wordpool_path}, using dummy words.")
        return [f"WORD{i}" for i in range(1000)]

    def get_trials(self):
        if self.pair_type == 'image':
            return self._get_image_trials()
        return self._get_word_trials()

    def _get_word_trials(self):
        n = min(self.n_images, len(self.dataset), len(self.wordpool))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:n]

        words = random.sample(self.wordpool, n)

        study_sequence = []
        pairs = []
        for i in range(n):
            img = self.dataset.get_image(selected_indices[i])
            word = words[i]
            study_sequence.append((img, word))
            pairs.append({
                "image": img,
                "word": word,
                "metadata": self.dataset.get_metadata(selected_indices[i])
            })

        test_phase = []
        test_indices = list(range(n))
        random.shuffle(test_indices)

        for i in test_indices:
            test_phase.append({
                "image": pairs[i]["image"],
                # The prompt lives in bench/tasks/pam_eval.py (it must follow the image).
                "target": pairs[i]["word"],
                "metadata": pairs[i]["metadata"]
            })

        return {
            "study_prompt": "Remember the word paired with each image.",
            "study_sequence": study_sequence,
            "test_phase": test_phase
        }

    def _get_image_trials(self):
        n_available = len(self.dataset)
        n = min(self.n_images, n_available // 2)

        indices = list(range(n_available))
        random.shuffle(indices)
        cue_indices = indices[:n]
        target_indices = indices[n:2 * n]

        pairs = []
        for i in range(n):
            pairs.append({
                "cue_image": self.dataset.get_image(cue_indices[i]),
                "target_image": self.dataset.get_image(target_indices[i]),
                "cue_metadata": self.dataset.get_metadata(cue_indices[i]),
                "target_metadata": self.dataset.get_metadata(target_indices[i]),
            })

        study_sequence = [
            {"images": [p["cue_image"], p["target_image"]], "pair_type": "pair"}
            for p in pairs
        ]

        test_phase = []
        test_indices = list(range(n))
        random.shuffle(test_indices)

        for i in test_indices:
            correct_img = pairs[i]["target_image"]
            foil_candidates = [j for j in range(n) if j != i]
            if foil_candidates:
                foil_idx = random.choice(foil_candidates)
                foil_img = pairs[foil_idx]["target_image"]
                foil_metadata = pairs[foil_idx]["target_metadata"]
            else:
                unused_indices = [idx for idx in indices[2 * n:] if idx not in {cue_indices[i], target_indices[i]}]
                if not unused_indices:
                    raise ValueError("Image-image paired associate 2-AFC requires at least one foil image.")
                foil_dataset_index = random.choice(unused_indices)
                foil_img = self.dataset.get_image(foil_dataset_index)
                foil_metadata = self.dataset.get_metadata(foil_dataset_index)

            if random.random() < 0.5:
                images = [correct_img, foil_img]
                target = 1
            else:
                images = [foil_img, correct_img]
                target = 2

            test_phase.append({
                "cue_image": pairs[i]["cue_image"],
                "images": images,
                "prompt": "Which of these two images was paired with the cue image? Reply with only the digit 1 or 2 and nothing else.",
                "target": target,
                "metadata": {
                    "cue": pairs[i]["cue_metadata"],
                    "correct_target": pairs[i]["target_metadata"],
                    "foil": foil_metadata,
                }
            })

        return {
            "study_prompt": "Remember which images were paired together.",
            "study_sequence": study_sequence,
            "test_phase": test_phase
        }


if __name__ == "__main__":
    task = PairedAssociateMemoryTask(n_images=5)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
