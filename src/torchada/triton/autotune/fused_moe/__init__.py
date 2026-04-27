import os


def set_default_moe_config_dir():
    default_path = os.path.dirname(os.path.realpath(__file__))

    if "SGLANG_MOE_CONFIG_DIR" not in os.environ:
        os.environ["SGLANG_MOE_CONFIG_DIR"] = default_path

    if "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = default_path
