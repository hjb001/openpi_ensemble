import os

from huggingface_hub import upload_folder, create_repo

HF_TOKEN = os.environ["HF_TOKEN"]

create_repo(
    repo_id="uuuhjb/sorting_clean_40000",
    repo_type="model",
    token=HF_TOKEN,
    exist_ok=True,
)

upload_folder(
    folder_path="app/checkpoints/sorting_clean_40000",
    repo_id="uuuhjb/sorting_clean_40000",
    token=HF_TOKEN,
)
