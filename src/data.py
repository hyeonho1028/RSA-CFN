import soundfile as sf
import torchaudio
from tqdm import tqdm

import pandas as pd
import numpy as np

def data_load(data_path='data/KEMDy20_v1_1/', use_wav=False):
    use_wav=True
    
    target_sampling_rate = 16000
    cols = ['Number', 'Wav_start', 'Wav_end', 'Segment ID', 'Emotion', 'Valence', 'Arousal']

    data = pd.DataFrame()
    for i in tqdm(range(1, 41)):
        
        i = str(i).zfill(2)
        
        tmp = pd.read_csv(data_path + f'annotation/Sess{i}_eval.csv').iloc[:, :7]
        tmp.columns = cols
        tmp = tmp.drop(0).reset_index(drop=True)
        
        s_id_ls = []
        s_id_wav_ls = []
        for s_id in tmp['Segment ID']:
            
            s = s_id.split('Sess')[1][:2]
            text_path = data_path + f'wav/Session{s}/{s_id}' + '.txt'
            wav_path = data_path + f'wav/Session{s}/{s_id}' + '.wav'
            
            text = pd.read_csv(text_path, encoding='cp949').columns[0]
            s_id_ls += [text]
            
            if use_wav:
                # try:
                    speech_array, sampling_rate = sf.read(wav_path)
                    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                    speech_array = resampler(speech_array).squeeze()
                    s_id_wav_ls += [[speech_array, sampling_rate]]
                # except:
                #     s_id_wav_ls += [[np.nan, np.nan]]

        else:
            tmp['text'] = s_id_ls
            tmp['speech_array'] = [i[0] for i in s_id_wav_ls]
            tmp['speech_sampling_rate'] = [i[1] for i in s_id_wav_ls]

        
        data = pd.concat([data, tmp])
        
    data = data.drop(columns='Number')
    return data