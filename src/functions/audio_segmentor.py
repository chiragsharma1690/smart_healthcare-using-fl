import pandas as pd
import librosa
import os
import glob
from pydub import AudioSegment
from tqdm import tqdm

def audio_segmentor():
    
    ad_segments_root_dir = "C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/segmentation/ad"
    cn_segments_root_dir = "C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/segmentation/cn"
    test_segments_root_dir = "C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/test-dist/segmentation"

    root_dir = "C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21"

    ad_audio_dataset_path = root_dir + "/diagnosis/train/audio/ad"
    cn_audio_dataset_path = root_dir + "/diagnosis/train/audio/cn"
    test_audio_dataset_path = root_dir + "/diagnosis/test-dist/audio"

    # for ad
    # partially working for ad for some systems
    # run in colab if not works
    for wav in glob.glob(ad_audio_dataset_path + "/*.wav"):

        real_file_name = wav.split('/')[-1]
        real_file_name = real_file_name.split('.')[0]
        real_file_name = real_file_name.split('\\')[1]

        df = pd.read_csv(ad_segments_root_dir + "/" + real_file_name + ".csv")

        try:
            a = AudioSegment.from_wav(wav)
        except:
            continue
        # a, sr = sf.read(wav)


        full_audio = a[0:0]
        print(real_file_name)
        for i, row in tqdm(df.iterrows()):

            if row['speaker'] == 'INV':
                continue
            
            full_audio += a[row['begin']: row['end']]
            print((i, row['begin'], row['end']))
        
        # sf.write("C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/audio/ad_segmented/" + real_file_name + ".wav", full_audio, 22050)
        
        full_audio.export(out_f="C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/audio/ad_segmented/" + real_file_name + ".wav", format="wav")
        os.remove(wav)
        print("\n\n")


    # for cn
    # one not working
    for wav in glob.glob(cn_audio_dataset_path + "/*.wav"):

        real_file_name = wav.split('/')[-1]
        real_file_name = real_file_name.split('.')[0]
        real_file_name = real_file_name.split('\\')[1]

        df = pd.read_csv(cn_segments_root_dir + "/" + real_file_name + ".csv")

        try:
            a = AudioSegment.from_wav(wav)
        except:
            continue
        # a, sr = sf.read(wav)


        full_audio = a[0:0]
        print(real_file_name)
        for i, row in tqdm(df.iterrows()):

            if row['speaker'] == 'INV':
                continue
            
            full_audio += a[row['begin']: row['end']]
            print((i, row['begin'], row['end']))
        
        # sf.write("C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/audio/ad_segmented/" + real_file_name + ".wav", full_audio, 22050)
        
        full_audio.export(out_f="C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/audio/cn_segmented/" + real_file_name + ".wav", format="wav")
        os.remove(wav)
        print("\n\n")
    
    
    # for test dataset
    # all is working
    for wav in glob.glob(test_audio_dataset_path + "/*.wav"):

        real_file_name = wav.split('/')[-1]
        real_file_name = real_file_name.split('.')[0]
        real_file_name = real_file_name.split('\\')[1]

        df = pd.read_csv(test_segments_root_dir + "/" + real_file_name + ".csv")

        try:
            a = AudioSegment.from_wav(wav)
        except:
            continue
        # a, sr = sf.read(wav)


        full_audio = a[0:0]
        print(real_file_name)
        for i, row in tqdm(df.iterrows()):

            if row['speaker'] == 'INV':
                continue
            
            full_audio += a[row['begin']: row['end']]
            print((i, row['begin'], row['end']))
        
        # sf.write("C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/train/audio/ad_segmented/" + real_file_name + ".wav", full_audio, 22050)
        
        full_audio.export(out_f="C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21/diagnosis/test-dist/audio_segmented/" + real_file_name + ".wav", format="wav")
        os.remove(wav)
        print("\n\n")

