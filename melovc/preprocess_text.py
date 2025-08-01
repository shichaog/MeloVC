import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
import os
import torch
import torchaudio
import numpy as np
from typing import Dict, Tuple
from text.symbols import symbols, num_languages, num_tones
from speechbrain.pretrained import EncoderClassifier

# --- Speaker Embedding 提取函数 ---
def get_speaker_embedding(encoder, wav_file_path):
    """
    提取单个音频文件的声纹向量。
    Args:
        wav_file_path (str): 音频文件的完整路径。
    Returns:
        list: 192维的声纹向量列表，如果文件不存在或处理失败则返回None。
    """
    if not os.path.exists(wav_file_path):
        print(f"警告：音频文件不存在，跳过：{wav_file_path}")
        return None
    try:
        signal, fs = torchaudio.load(wav_file_path)
        # SpeechBrain模型通常期望16kHz的采样率，如果你的音频不是，可能需要重采样
        if fs != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = transform(signal)
        
        # 确保信号是单声道 (如果需要，SpeechBrain模型可能对多声道有特定要求)
        if signal.ndim > 1 and signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        with torch.no_grad():
            # D-vector/X-vector, shape: (1, 1, 192)
            embedding = encoder.encode_batch(signal) 
            # 压缩维度，使其变为 (192,)
            embedding = embedding.squeeze()
            embedding = torch.round(embedding * 100) / 100 
        return embedding.tolist() # 将Tensor转换为Python列表以便JSON序列化
    except Exception as e:
        print(f"处理音频文件失败：{wav_file_path}，错误：{e}")
        return None

@click.command()
@click.option(
    "--metadata",
    default="/mnt/bn/automastering-396caa4f/MeloVC/melovc/data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="/mnt/bn/automastering-396caa4f/MeloVC/melovc/configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--max-val-total", default=1)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    max_val_total: int,
    clean: bool,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    try:
        encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             run_opts={"device": "cuda"}) # 关键修改：指定设备为 "cuda" 或 "cpu")
        print("speechbrain speaker embedding 模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保网络连接正常，或者尝试手动下载模型文件。")
        exit()

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        new_symbols = []
        for line in tqdm(open(metadata, encoding="utf-8").readlines()):
            try:
                parts = line.strip().split("|")
                utt, language, text = parts[0], parts[1], parts[2]
                embedding = get_speaker_embedding(encoder, utt)
                embedding = ' '.join([f"{x:.3f}" for x in embedding])

                norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda:0')
                for ph in phones:
                    if ph not in symbols and ph not in new_symbols:
                        new_symbols.append(ph)
                        print('update!, now symbols:')
                        print(new_symbols)
                        with open(f'{language}_symbol.txt', 'w') as f:
                            f.write(f'{new_symbols}')

                assert len(phones) == len(tones)
                assert len(phones) == sum(word2ph)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        language,
                        norm_text,
                        embedding,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                bert_path = utt.replace(".wav", ".bert.pt")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert.cpu(), bert_path)
            except Exception as error:
                print("err!", line, error)

        out_file.close()
        metadata = cleaned_path

    with open(metadata, encoding="utf-8") as f:
        lines = f.readlines()
            

    # 划分训练集和验证集
    val_list = lines[:max_val_total]
    train_list = lines[max_val_total:]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path

    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
