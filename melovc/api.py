import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model

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

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=False,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            num_tones=num_tones,
            num_languages=num_languages,
            use_vc=True,
            hps=hps,
            sample_rate = hps.data.sampling_rate,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        # The weights might be directly in the checkpoint or nested under a key like 'model' or 'model_state_dict'
        if 'model_state_dict' in checkpoint_dict:
            state_dict = checkpoint_dict['model_state_dict']
        elif 'model' in checkpoint_dict:
            state_dict = checkpoint_dict['model']
        else:
            state_dict = checkpoint_dict

        # Print all the layer names (keys) in the checkpoint
        print("Layers found in the checkpoint file:")
        for key in state_dict.keys():
            print(key)
            
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, ref_audio_path, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)

        device = self.device

        if ref_audio_path is None:
            speaker_embed = torch.zeros(1, 192).to(device)
            print("++++", speaker_embed.shape)
        else:
            try:
                encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             run_opts={"device": "cuda"}) # 关键修改：指定设备为 "cuda" 或 "cpu")
                print("speechbrain speaker embedding 模型加载成功！")
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("请确保网络连接正常，或者尝试手动下载模型文件。")
                exit()
            speaker_embed = get_speaker_embedding(encoder, ref_audio_path)
            speaker_embed = torch.FloatTensor([speaker_embed]).to(device)
            print("++++1", speaker_embed.shape)

        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones

                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        f0_external=None,
                        tone=tones,
                        language=lang_ids,
                        speaker_embed=speaker_embed,
                        bert=bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, x_tst_lengths
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
