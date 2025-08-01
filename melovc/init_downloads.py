

if __name__ == '__main__':

    from melovc.api import TTS
    device = 'auto'
    models = {
        'EN': TTS(language='EN', device=device),
        'ZH': TTS(language='ZH', device=device),
    }
