import torch
import os
from . import utils
from cached_path import cached_path
from huggingface_hub import hf_hub_download


PRETRAINED_MODELS_REPO_ID = 'shichaog/MeloVC'

PRETRAINED_MODELS = {
        'G.pth': 'G.pth',
        'D.pth': 'D.pth',
        'DUR.pth': 'DUR.pth',
        }

LANG_TO_HF_REPO_ID = {
    'EN': 'shichaog/MeloVC',
    'ZH': 'shichaog/MeloVC',
}

def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="G.pth")
    return torch.load(ckpt_path, map_location=device)

def load_pretrain_model():
    downloaded_paths = [
        hf_hub_download(
            repo_id=PRETRAINED_MODELS_REPO_ID,
            filename=model_filename
        )
        for model_filename in PRETRAINED_MODELS.values()
    ]
    return downloaded_paths
