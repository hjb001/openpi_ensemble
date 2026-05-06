import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="uuuhjb/roll_v3_47999",
    token=os.environ["HF_TOKEN"],
    local_dir="app/checkpoints/roll_v3_47999",
)
