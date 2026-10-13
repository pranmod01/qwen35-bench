import math
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import BradyDataset, DirectoryDataset, ImageNetConceptsDataset, ThingsDataset


def lag_bin(serial_lag):
    if serial_lag <= 1:
        return "lag_1"
    if serial_lag <= 3:
        return "lag_2_3"
    if serial_lag <= 10:
        return "lag_4_10"
    if serial_lag <= 100:
        return "lag_11_100"
    return "lag_gt_100"


def balanced_sample_by_lag(pair_indices, n_tests):
    groups = {}
    for left_idx, right_idx in pair_indices:
        groups.setdefault(lag_bin(abs(left_idx - right_idx)), []).append((left_idx, right_idx))
    for group in groups.values():
        random.shuffle(group)

    selected = []
    bins = sorted(groups)
    while len(selected) < n_tests:
        progressed = False
        round_bins = bins[:]
        random.shuffle(round_bins)
        for bin_id in round_bins:
            if len(selected) >= n_tests:
                break
            if groups[bin_id]:
                selected.append(groups[bin_id].pop())
                progressed = True
        if not progressed:
            break
    random.shuffle(selected)
    return selected


class SerialOrderMemoryBase:
    def __init__(self, dataset_name="Brady2008", n_images=20, image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks", exemplars_per_concept=1):
        self.dataset_name = dataset_name
        self.n_images = n_images
        self.image_dir = image_dir
        self.source = source
        self.repo_id = repo_id
        # THINGS has 1,854 concepts, so one photo each caps the list there. k > 1 draws k photos per
        # concept and shuffles them into one list; test pairs never compare two photos of one concept.
        if exemplars_per_concept > 1 and dataset_name != "things":
            raise ValueError("exemplars_per_concept > 1 is only defined for THINGS")
        self.exemplars_per_concept = exemplars_per_concept
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.image_dir:
            return DirectoryDataset(self.image_dir)
        if self.dataset_name == "Brady2008":
            return BradyDataset(type="Objects", source=self.source, repo_id=self.repo_id)
        if self.dataset_name == "imagenet":
            dataset = ImageNetConceptsDataset()
            # _sample_study_items would otherwise shorten the list silently.
            if self.n_images > len(dataset):
                raise ValueError(f"imagenet has {len(dataset)} concepts; asked for {self.n_images}")
            return dataset
        k = self.exemplars_per_concept
        if k > 1:
            return ThingsDataset(n_categories=math.ceil(self.n_images / k), exemplars_per_category=k)
        return ThingsDataset(n_categories=self.n_images)

    def _sample_study_items(self):
        k = self.exemplars_per_concept
        # (dataset index, exemplar index) units; with k == 1 this is the original one-per-item draw.
        units = [(i, e) for i in range(len(self.dataset)) for e in range(k)] if k > 1 \
            else list(range(len(self.dataset)))
        n = min(self.n_images, len(units))
        random.shuffle(units)
        selected_indices = units[:n]

        study_sequence = []
        study_items = []
        for serial_position, unit in enumerate(selected_indices, start=1):
            dataset_index, exemplar = unit if k > 1 else (unit, None)
            image = self.dataset.get_image(dataset_index) if k == 1 else self.dataset.get_image(dataset_index, exemplar)
            metadata = {
                **(self.dataset.get_metadata(dataset_index) if k == 1 else self.dataset.get_metadata(dataset_index, exemplar)),
                "serial_position": serial_position,
            }
            study_sequence.append(image)
            study_items.append(
                {
                    "dataset_index": dataset_index,
                    "serial_position": serial_position,
                    "image": image,
                    "metadata": metadata,
                }
            )
        return study_sequence, study_items


class SerialOrderMemoryTask(SerialOrderMemoryBase):
    def get_trials(self):
        study_sequence, study_items = self._sample_study_items()
        n = len(study_items)

        test_indices = list(range(n))
        random.shuffle(test_indices)

        test_phase = []
        for item_index in test_indices:
            item = study_items[item_index]
            test_phase.append(
                {
                    "image": item["image"],
                    "prompt": f"What position in the sequence did this image appear (1-{n})? Reply with only the position number and nothing else.",
                    "target": item["serial_position"],
                    "metadata": item["metadata"],
                }
            )

        return {
            "study_prompt": f"Here is a sequence of {n} images to remember in order.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


class AFCSerialOrderMemoryTask(SerialOrderMemoryBase):
    def __init__(self, dataset_name="Brady2008", n_images=20, n_tests=None, image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks", exemplars_per_concept=1):
        super().__init__(
            dataset_name=dataset_name,
            n_images=n_images,
            image_dir=image_dir,
            source=source,
            repo_id=repo_id,
            exemplars_per_concept=exemplars_per_concept,
        )
        self.n_tests = n_tests

    def get_trials(self):
        study_sequence, study_items = self._sample_study_items()
        n = len(study_items)

        pair_indices = list(combinations(range(n), 2))
        if self.exemplars_per_concept > 1:
            category = [item["metadata"]["category"] for item in study_items]
            pair_indices = [(a, b) for a, b in pair_indices if category[a] != category[b]]
        max_pairs = len(pair_indices)
        n_tests = n if self.n_tests is None else self.n_tests
        n_tests = min(n_tests, max_pairs)
        pair_indices = balanced_sample_by_lag(pair_indices, n_tests)

        test_phase = []
        for left_idx, right_idx in pair_indices:
            first_item = study_items[left_idx]
            second_item = study_items[right_idx]
            pair = [first_item, second_item]
            random.shuffle(pair)

            target = 1 if pair[0]["serial_position"] < pair[1]["serial_position"] else 2
            serial_lag = abs(first_item["serial_position"] - second_item["serial_position"])
            distance = serial_lag - 1

            test_phase.append(
                {
                    "images": [pair[0]["image"], pair[1]["image"]],
                    # The prompt lives in bench/tasks/serial_order_eval.py (it must follow the images).
                    "target": target,
                    "metadata": {
                        "first_serial_position": min(first_item["serial_position"], second_item["serial_position"]),
                        "second_serial_position": max(first_item["serial_position"], second_item["serial_position"]),
                        "serial_lag": serial_lag,
                        "lag_bin": lag_bin(serial_lag),
                        "distance": distance,
                        "left_image_serial_position": pair[0]["serial_position"],
                        "right_image_serial_position": pair[1]["serial_position"],
                    },
                }
            )

        return {
            "study_prompt": f"Here is a sequence of {n} images to remember in order.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


__all__ = ["SerialOrderMemoryTask", "AFCSerialOrderMemoryTask"]
