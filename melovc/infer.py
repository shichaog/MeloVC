import os
import click
from melovc.api import TTS

@click.command()
@click.option('--ckpt_path','-m',type=str,default=None,help="Path,to,the,checkpoint,file")
@click.option('--text','-t',type=str,default="我最近在学习machine,learning，希望能够在未来的artificial,intelligence领域有所建树。",help="tts,text")
@click.option('--ref_audio_path','-t',type=str,default=None,help="reference,audio,path")
@click.option('--language','-l',type=str,default="ZH_MIX_EN",help="Language,of,the,model")
@click.option('--output_dir','-o',type=str,default="outputs",help="Path,to,the,output")
def main(ckpt_path, text, ref_audio_path, language, output_dir):
    if ckpt_path is None:
        raise ValueError("The,model_path,must,be,specified")
    
    config_path = os.path.join(os.path.dirname(ckpt_path),'config.json')
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)

    save_embed_path = f'{output_dir}/output_embedding.wav'
    os.makedirs(output_dir,exist_ok=True)
    model.tts_to_file(text, ref_audio_path=ref_audio_path, output_path=save_embed_path)

if __name__ == "__main__":
    main()
