import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import os
import glob
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def feature_extractor_train2021(df):

    root_dir = "C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSSo21"
    train_audio_dataset_path = root_dir + "/diagnosis/train/audio"
    extracted_features = []

    for i, row in tqdm(df.iterrows()):
        
        file_name = os.path.join(os.path.abspath(train_audio_dataset_path), str(row["dx"]) + '_segmented/', str(row["adressfname"]) + '.wav')
        mmse_scores = row["mmse"]
        class_folder = row["dx"]
        if class_folder == 'ad':
            class_value = 1
        else:
            class_value = 0

        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

        # print(wav)

        extracted_features.append([mfccs_scaled_features, class_value, mmse_scores])

    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature','class','mmse_scores'])
    # print(extracted_features_df.shape)
    # extracted_features_df.head()

    ### Split the dataset into independent and dependent dataset
    X=np.array(extracted_features_df['feature'].tolist())
    y=np.array(extracted_features_df['class'].tolist())

    return X, y

def feature_extractor_test(df):

    pass

def feature_extractor_train2020():

    ad_train_path = 'C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSS-IS2020-data/train/Normalised_audio-chunks/cd'

    extracted_features_cd = []
    class_value = 1
    for wav in glob.glob(ad_train_path + '/' + '*.wav'):
        # print(wav)
        audio, sample_rate = librosa.load(wav, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

        extracted_features_cd.append([mfccs_scaled_features, class_value])
    
    cc_train_path = 'C:/Users/chira/Downloads/Federated Learning Project/my_code_exp/ADReSS-IS2020-data/train/Normalised_audio-chunks/cc'

    extracted_features_cc = []
    class_value = 0
    for wav in glob.glob(cc_train_path + '/' + '*.wav'):
        # print(wav)
        audio, sample_rate = librosa.load(wav, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

        extracted_features_cc.append([mfccs_scaled_features, class_value])
    
    extracted_features_df2 = pd.DataFrame(extracted_features_cc, columns=['feature','class'])
    extracted_features_df1 = pd.DataFrame(extracted_features_cd, columns=['feature','class'])
    extracted_features_df = pd.concat([extracted_features_df1, extracted_features_df2])


    extracted_features_df_shuffled = shuffle(extracted_features_df)
    extracted_features_df_shuffled.reset_index(drop=True, inplace=True)

    ### Split the dataset into independent and dependent dataset
    X=np.array(extracted_features_df_shuffled['feature'].tolist())
    y=np.array(extracted_features_df_shuffled['class'].tolist())

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    X_scaled

    return X_scaled, y
