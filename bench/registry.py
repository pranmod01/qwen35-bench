"""Task name -> evaluation wrapper.

A wrapper provides NAME, expand_conditions(task_cfg) and run(evaluator, condition, seed), and optionally
max_n_images(condition) / min_n_images(condition), which the runner checks before loading any model.
"""
from bench.tasks import (afc_recognition_eval, associative_inference_eval, color_memory_eval,
                        mst_eval, pam_eval, serial_order_eval)

ADAPTERS = {
    module.NAME: module
    for module in (afc_recognition_eval, associative_inference_eval, color_memory_eval,
                   mst_eval, pam_eval, serial_order_eval)
}


def get_adapter(name):
    if name not in ADAPTERS:
        raise ValueError(f"unknown task {name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]
