import torch
import os
from . import utils
from cached_path import cached_path
from huggingface_hub import hf_hub_download


PRETRAINED_MODELS_REPO_ID = 'shichaog/MeloVC'

PRETRAINED_MODELS = {
        'G.pth': 'G_V2.pth',
        'D.pth': 'D_V2.pth',
        'DUR.pth': 'DUR_V2.pth',
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

def load_pretrain_model(device:str, ckpt_path: str=None):
    if ckpt_path is None:
        local_dir = 'pretrained_melovc_models'
        print(f"未提供 ckpt_path，将使用默认路径 '{local_dir}' 检查和存放模型。")
    else:
        local_dir = ckpt_path

    # 2. 确保目录存在
    os.makedirs(local_dir, exist_ok=True)

    model_paths = {}
    all_files_ready = True

    try:
        for key, filename in PRETRAINED_MODELS.items():
            expected_path = os.path.join(local_dir, filename)

            if not os.path.exists(expected_path):
                print(f"本地未找到模型文件 '{filename}'，正在从 Hugging Face Hub 下载...")
                try:
                    # 使用 hf_hub_download 下载，它会自动处理缓存和文件放置
                    downloaded_path = hf_hub_download(
                        repo_id=PRETRAINED_MODELS_REPO_ID,
                        filename=filename,
                        local_dir=local_dir,
                        # 建议设置为 False，避免在某些系统上因符号链接产生问题
                        local_dir_use_symlinks=False 
                    )
                    model_paths[key] = downloaded_path
                    print(f"'{filename}' 下载完成。")
                except Exception as e:
                    print(f"下载 '{filename}' 失败: {e}")
                    all_files_ready = False
                    break # 如果一个文件下载失败，就没必要继续了
            else:
                print(f"在本地找到模型文件: '{expected_path}'")
                model_paths[key] = expected_path
        if not all_files_ready:
            print("部分模型文件下载失败，无法继续。")
            return None, None, None
        
        return model_paths['G.pth'], model_paths['D.pth'], model_paths['DUR.pth']
        
    except Exception as e:
        print(f"处理模型文件时发生严重错误: {e}")
        return None, None, None
