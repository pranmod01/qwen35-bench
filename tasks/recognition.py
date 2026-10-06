import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import BradyDataset, CuedRecallImageNetDataset, DirectoryDataset, ThingsDataset


CUED_RECALL_IMAGENET = "cued-recall-imagenet"


class RecognitionTaskBase:
    def __init__(self, dataset_name="Brady2008", n_images=20, image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks"):
        self.dataset_name = dataset_name
        self.n_images = n_images
        self.image_dir = image_dir
        self.source = source
        self.repo_id = repo_id

    def _load_recognition_dataset(self, exemplars_per_category=1):
        if self.image_dir:
            return DirectoryDataset(self.image_dir)
        if self.dataset_name == "Brady2008":
            return BradyDataset(type="Objects", source=self.source, repo_id=self.repo_id)
        return ThingsDataset(
            n_categories=self.n_images,
            exemplars_per_category=exemplars_per_category,
        )


class ContinuousRecognitionTask(RecognitionTaskBase):
    def __init__(self, dataset_name="Brady2008", n_images=50, n_trials=None,
                 min_delay=2, max_delay=15, p_old=0.5, image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks"):
        super().__init__(
            dataset_name=dataset_name,
            n_images=n_images,
            image_dir=image_dir,
            source=source,
            repo_id=repo_id,
        )
        self.n_trials = n_trials
        # For very small n, delay constraints can't be satisfied — drop them so
        # every image is immediately eligible to repeat after first presentation.
        if n_images < min_delay + 2:
            min_delay = 0
            max_delay = n_images
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.p_old = p_old
        self.dataset = self._load_recognition_dataset()

    def generate_sequence(self):
        n_unique = min(self.n_images, len(self.dataset))
        if self.n_trials is not None:
            total_trials = self.n_trials
            n_old_needed = int(total_trials * self.p_old)
            n_unique = min(n_unique, total_trials - n_old_needed)
        else:
            total_trials = int(n_unique / (1 - self.p_old))
            n_old_needed = total_trials - n_unique

        sequence = []
        new_indices = list(range(n_unique))
        random.shuffle(new_indices)

        waiting_room = []  # (img_idx, introduction_time)

        current_time = 0
        while len(sequence) < total_trials:
            eligible_to_repeat = [
                (idx, img_idx, t_intro)
                for idx, (img_idx, t_intro) in enumerate(waiting_room)
                if (current_time - t_intro - 1) >= self.min_delay
            ]
            must_repeat = [
                (idx, img_idx, t_intro)
                for idx, (img_idx, t_intro) in enumerate(waiting_room)
                if (current_time - t_intro - 1) >= self.max_delay
            ]

            show_old = False
            if must_repeat:
                show_old = True
            elif not new_indices:
                show_old = True
            elif eligible_to_repeat and n_old_needed > 0 and random.random() < self.p_old:
                show_old = True

            if show_old and eligible_to_repeat:
                if must_repeat:
                    idx_in_waiting, img_idx, t_intro = must_repeat[0]
                else:
                    idx_in_waiting, img_idx, t_intro = random.choice(eligible_to_repeat)

                waiting_room.pop(idx_in_waiting)
                sequence.append(
                    {
                        "image_idx": img_idx,
                        "target": 1,
                        "delay": current_time - t_intro - 1,
                    }
                )
                n_old_needed -= 1
            elif new_indices:
                img_idx = new_indices.pop(0)
                waiting_room.append((img_idx, current_time))
                sequence.append(
                    {
                        "image_idx": img_idx,
                        "target": 0,
                        "delay": None,
                    }
                )
            elif waiting_room:
                # All new images exhausted; wait until oldest waiting image is
                # eligible (respects min_delay) before repeating.
                img_idx, t_intro = waiting_room[0]
                delay = current_time - t_intro - 1
                if delay >= self.min_delay:
                    waiting_room.pop(0)
                    sequence.append(
                        {
                            "image_idx": img_idx,
                            "target": 1,
                            "delay": delay,
                        }
                    )
                    n_old_needed -= 1
                # else: not yet eligible — advance current_time and retry
            else:
                break

            current_time += 1

        return sequence

    def get_trials(self):
        sequence = self.generate_sequence()
        trials = []
        for item in sequence:
            trials.append(
                {
                    "image": self.dataset.get_image(item["image_idx"]),
                    "prompt": "Has this image already appeared in the sequence? Reply with only 'yes' or 'no' and nothing else.",
                    "target": item["target"],
                    "metadata": {
                        **self.dataset.get_metadata(item["image_idx"]),
                        "delay": item["delay"],
                    },
                }
            )
        return trials


class AFCRecognitionTask(RecognitionTaskBase):
    def __init__(self, dataset_name="Brady2008", n_images=20, foil_type="all", image_dir=None,
                 source="local", repo_id="chrisiyer/vision-memory-tasks", single_foil=False):
        super().__init__(
            dataset_name=dataset_name,
            n_images=n_images,
            image_dir=image_dir,
            source=source,
            repo_id=repo_id,
        )
        self.foil_type = foil_type
        # Novel foils only. By default every studied item gets its own foil (2n distinct items), but
        # the bench tests one item per episode, so that halves the largest study list for nothing.
        # single_foil draws n + 1 items: n studied, one foil, attached to one random studied item.
        if single_foil and foil_type != "novel":
            raise ValueError("single_foil is only defined for novel foils")
        self.single_foil = single_foil

        if image_dir:
            self.dataset = self._load_recognition_dataset()
        elif dataset_name == "things":
            if foil_type == "state":
                raise ValueError("State foils not supported for THINGS dataset.")
            exemplars = 2 if foil_type in ["exemplar", "all"] else 1
            # Novel pairs need 2 distinct categories each; 'all' splits half novel / half exemplar.
            if foil_type == "novel":
                n_cats = n_images + 1 if single_foil else n_images * 2
            elif foil_type == "all":
                n_cats = (n_images * 3 + 1) // 2
            else:
                n_cats = n_images
            self.dataset = ThingsDataset(n_categories=n_cats, exemplars_per_category=exemplars)
        elif dataset_name == CUED_RECALL_IMAGENET:
            if foil_type == "state":
                raise ValueError(
                    "State foils not supported for cued-recall-imagenet (no paired state images)."
                )
            exemplars = 2 if foil_type in ["exemplar", "all"] else 1
            if foil_type == "novel":
                n_cats = n_images * 2
            elif foil_type == "all":
                n_cats = (n_images * 3 + 1) // 2
            else:
                n_cats = n_images
            self.dataset = CuedRecallImageNetDataset(
                n_categories=n_cats, exemplars_per_category=exemplars
            )
        else:
            self.dataset = BradyDataset(type="Objects", source=source, repo_id=repo_id)

    def _get_pairs(self, foil_type, n):
        pairs = []

        if self.image_dir:
            # Directory dataset only supports novel foils
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            for i in range(0, min(n * 2, len(indices) - 1), 2):
                pairs.append(
                    {
                        "original": self.dataset.get_image(indices[i]),
                        "foil": self.dataset.get_image(indices[i + 1]),
                        "type": "novel",
                    }
                )
            return pairs

        if self.single_foil and self.dataset_name in ("things", "Brady2008"):
            if self.dataset_name == "things":
                ds, get = self.dataset, (lambda i: self.dataset.get_image(i, 0))
            else:
                ds = BradyDataset(type="Objects", source=self.source, repo_id=self.repo_id)
                get = ds.get_image
            indices = list(range(len(ds)))
            if len(indices) < n + 1:
                raise ValueError(f"single-foil novel pairs need {n + 1} distinct items, have {len(indices)}")
            random.shuffle(indices)
            # Only pairs with a foil become test trials (see get_trials), so the one tested item sits at
            # a random study position once the pairs are shuffled. The default mode does *not* do this:
            # its test_phase follows study order and the bench scores test_phase[0], so it always tests
            # the first studied item (found 2026-10-04).
            return [{"original": get(indices[i]), "foil": get(indices[n]) if i == 0 else None, "type": "novel"}
                    for i in range(n)]

        if self.dataset_name in ("things", CUED_RECALL_IMAGENET):
            if foil_type == "novel" or foil_type == "all":
                n_novel = n if foil_type == "novel" else n // 2
                n_exemplar = n - n_novel

                indices = list(range(len(self.dataset)))
                if len(indices) < n_novel * 2:
                    raise ValueError(
                        f"THINGS novel pairs require {n_novel * 2} distinct categories "
                        f"but only {len(indices)} are available (n_images={n}, foil_type={foil_type})."
                    )
                random.shuffle(indices)

                for i in range(0, n_novel * 2, 2):
                    pairs.append(
                        {
                            "original": self.dataset.get_image(indices[i], 0),
                            "foil": self.dataset.get_image(indices[i + 1], 0),
                            "type": "novel",
                        }
                    )

                start_idx = n_novel * 2
                for i in range(start_idx, start_idx + n_exemplar):
                    if i < len(indices):
                        idx = indices[i]
                        pairs.append(
                            {
                                "original": self.dataset.get_image(idx, 0),
                                "foil": self.dataset.get_image(idx, 1),
                                "type": "exemplar",
                            }
                        )
            elif foil_type == "exemplar":
                indices = list(range(len(self.dataset)))
                random.shuffle(indices)
                for i in range(min(n, len(indices))):
                    idx = indices[i]
                    pairs.append(
                        {
                            "original": self.dataset.get_image(idx, 0),
                            "foil": self.dataset.get_image(idx, 1),
                            "type": "exemplar",
                        }
                    )
            return pairs

        if foil_type == "novel":
            obj_ds = BradyDataset(type="Objects", source=self.source, repo_id=self.repo_id)
            indices = list(range(len(obj_ds)))
            if len(indices) < n * 2:
                raise ValueError(
                    f"Brady novel pairs require {n * 2} distinct objects "
                    f"but only {len(indices)} are available (n_images={n})."
                )
            random.shuffle(indices)
            for i in range(0, n * 2, 2):
                pairs.append(
                    {
                        "original": obj_ds.get_image(indices[i]),
                        "foil": obj_ds.get_image(indices[i + 1]),
                        "type": "novel",
                    }
                )
        elif foil_type == "exemplar" or foil_type == "state":
            ds = BradyDataset(
                type="Exemplar" if foil_type == "exemplar" else "State",
                source=self.source,
                repo_id=self.repo_id,
            )
            # Pair order on disk is alphabetical, so taking the first n made every n=1 episode use
            # the same pair and every n=5 the same five (cf. the THINGS sampler fix in src/stimuli.py).
            pair_indices = list(range(len(ds.pair_paths)))
            random.shuffle(pair_indices)
            for i in pair_indices[:min(n, len(ds.pair_paths))]:
                original, foil = ds.get_pair(i)
                if random.random() < 0.5:
                    original, foil = foil, original
                pairs.append(
                    {
                        "original": original,
                        "foil": foil,
                        "type": foil_type,
                    }
                )
        elif foil_type == "all":
            n3 = n // 3
            pairs += self._get_pairs("novel", n3)
            pairs += self._get_pairs("exemplar", n3)
            pairs += self._get_pairs("state", n - 2 * n3)

        return pairs

    def get_trials(self):
        pairs = self._get_pairs(self.foil_type, self.n_images)
        random.shuffle(pairs)

        study_sequence = [pair["original"] for pair in pairs]

        test_phase = []
        for pair in pairs:
            if pair["foil"] is None:  # single_foil: studied, never tested
                continue
            correct_img = pair["original"]
            foil_img = pair["foil"]
            if random.random() < 0.5:
                images = [correct_img, foil_img]
                target = 1
            else:
                images = [foil_img, correct_img]
                target = 2

            test_phase.append(
                {
                    "images": images,
                    # The prompt lives in bench/tasks/afc_recognition_eval.py (it must follow the images).
                    "target": target,
                    "type": pair["type"],
                }
            )

        return {
            "study_prompt": "Here is a sequence of images to remember.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


__all__ = ["RecognitionTaskBase", "ContinuousRecognitionTask", "AFCRecognitionTask"]
