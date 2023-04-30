import os
import random
import torch
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoConfig, AutoTokenizer

from transformers import XLMRobertaConfig, XLMRobertaTokenizer

from src.data import data_load
from src.loader import Custom_Dataset, DataCollatorCTCWithPadding
from src.trainer import training

import warnings
warnings.filterwarnings('ignore')

def seed_everything(SEED):
	random.seed(SEED)
	os.environ['PYTHONHASHSEED'] = str(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# ------------------------
#  Arguments
# ------------------------
parser = argparse.ArgumentParser(description='exp_KEM')

# Base settings
parser.add_argument('--data_path', type=str, default='data/KEMDy20_v1_1/', help='Path to data directory')
parser.add_argument('--output_dir', type=str, default='models/', help='Path to output directory')
parser.add_argument('--ver', type=str, default='baseline', help='Model version')
parser.add_argument('--audio_max_lens', type=int, default=96000, help='Number of Audio max lengths')
parser.add_argument('--text_max_lens', type=int, default=256, help='Number of Text max lengths')
parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
parser.add_argument('--seed', type=int, default=2023, help='Random seed')

# Device settings
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='Device type (cpu, cuda, mps)')
parser.add_argument('--use_amp', type=bool, default=True, help='Use Automatic Mixed Precision for training', action=argparse.BooleanOptionalAction)

# Label transformation
parser.add_argument('--label_transform', type=bool, default=False, help='Perform label transformation', action=argparse.BooleanOptionalAction)
parser.add_argument('--batch_weighted_sampler', type=bool, default=False, help='Use batch weighted sampler for training', action=argparse.BooleanOptionalAction)

# Training setting
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')

# Training mode
parser.add_argument('--use_wav', type=bool, default=False, help='Use wav2vec feature for training', action=argparse.BooleanOptionalAction)
parser.add_argument('--use_concat', type=bool, default=False, help='Concatenate modalities for training', action=argparse.BooleanOptionalAction)
parser.add_argument('--multimodal_method', type=str, default='early_fusion', 
                    choices=['mlp_mixer',
                             'early_fusion', 'late_fusion',  'stack', 
                             'residual', 'rsa', 'rsa_cfn',
                             'hybrid_fusion'], help='Multimodal method for training')

# Backbones
parser.add_argument('--wav_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self', help='Pre-trained wav2vec model')
parser.add_argument('--pooling_mode', type=str, default='mean', help='Pooling mode for feature extraction')
parser.add_argument('--rnn_model', type=str, default='klue/roberta-base', choices=['xlm-roberta-base', 'xlm-roberta-large', 'klue/roberta-base', 'klue/roberta-large'], help='Pre-trained RNN model')

def main(args):

    if args.use_wav:
        if not args.use_concat:
            args.lr = 1e-4
            
    if args.use_concat:
        args.use_wav = True
        args.batch_size = 6
        args.val_batch_size = 16
            
    if args.multimodal_method in ['rsa_cfn', 'hybrid_fusion']:
        args.batch_size = 4
        args.val_batch_size = 16

    print('Load Merge Data')
    data = data_load(data_path=args.data_path)
    data['Emotion'] = data['Emotion'].apply(lambda x: x.replace('disqust', 'disgust'))
    print('Complete Merge Dataset')

    label_dict = {'neutral': 0, 'happy': 1, 'angry': 2, 'surprise': 3, 'sad': 4, 'fear': 5, 'disgust': 6}
    subset_data = data.copy()
    subset_data['Emotion'] = subset_data['Emotion'].map(label_dict)
    subset_data = subset_data.dropna(subset=['Emotion'])
    print(subset_data)

    # data split
    df_train, df_valid = train_test_split(
                                        subset_data,
                                        test_size=0.2,
                                        random_state=42,
                                        shuffle=True,
                                        stratify=subset_data['Emotion'],
                                        )

    # set label transform
    if args.label_transform:
        print('Apply Label transform')
        modify_data = data[data['Emotion'].apply(lambda x: (True if len(x.split(';'))==2 else False) & ('neutral' in  x.split(';')))]
        modify_data['Emotion'] = modify_data['Emotion'].apply(lambda x: x.replace('neutral', '').replace(';', ''))
        modify_data['Emotion'] = modify_data['Emotion'].map(label_dict)
        df_train = pd.concat([df_train, modify_data], axis=0)

    df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)
    print(df_train.shape, df_valid.shape)

    # set batch weighted sampler
    if args.batch_weighted_sampler:
        print('Apply batch weighted sampler')
        y_train = df_train['Emotion']
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        weight_dict = dict(zip(np.unique(y_train), weight))
        args.samples_weight = np.array([weight_dict[t] for t in y_train])


    print(data['speech_array'].apply(len).max(), data['speech_array'].apply(len).min())
    # max_lens = data['speech_array'].apply(len).max()
    # 563472

    # wav
    args.wav_config = AutoConfig.from_pretrained(args.wav_model, num_labels=len(label_dict), finetuning_task='wav2vec2_clf')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav_model)
    args.data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True, max_length=args.audio_max_lens)

    # rnn
    if 'xlm' in args.rnn_model:
        args.rnn_config = XLMRobertaConfig.from_pretrained(args.rnn_model)
        args.rnn_config.hidden_dropout_prob = 0
        args.rnn_config.attention_probs_dropout_prob = 0
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.rnn_model)
    else:
        args.rnn_config = AutoConfig.from_pretrained(args.rnn_model)
        args.rnn_config.hidden_dropout_prob = 0
        args.rnn_config.attention_probs_dropout_prob = 0
        tokenizer = AutoTokenizer.from_pretrained(args.rnn_model)

    # dataset
    args.train_dataset = Custom_Dataset(args, df_train, tokenizer, feature_extractor)
    args.valid_dataset = Custom_Dataset(args, df_valid, tokenizer, feature_extractor)

    # training - evaluate - model save
    training(args)

if __name__=='__main__':
    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
