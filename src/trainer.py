from .models import Wav2Vec2ForSpeechClassification, xlm_RoBertaBase, MultiModel

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler

from sklearn.metrics import f1_score

# sharing model forward
def sharing_step(args, model, batch):
    attn_w = None
    if args.use_concat:
        if args.multimodal_method=='rsa':
            pred, attn_w = model(batch['input_values'], 
                                 batch['tokenized_input_ids'], 
                                 batch['tokenized_attention_mask'], 
                                 batch['tokenized_token_type_ids'])
        else:
            pred = model(batch['input_values'], 
                        batch['tokenized_input_ids'], 
                        batch['tokenized_attention_mask'], 
                        batch['tokenized_token_type_ids'])
    else:
        if args.use_wav:
            pred = model(batch['input_values'])
        else:
            pred = model(batch['tokenized_input_ids'], 
                         batch['tokenized_attention_mask'], 
                         batch['tokenized_token_type_ids'])
    
    return pred, attn_w


# training function
def training(args):
    '''
    [description]
    function for training
    After training, save model and bin-count of result.

    [args]
    args: argparse object
        args must have following attributes:
            - train_dataset
            - valid_dataset

    [return]
    None
    '''
    
    if args.batch_weighted_sampler:
        samples_weight = torch.from_numpy(args.samples_weight)
        train_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    else:
        train_sampler = RandomSampler(args.train_dataset)
            
    train_dataloader = DataLoader(
                            args.train_dataset,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False,
                            sampler = train_sampler,
                            collate_fn=args.data_collator,
                            drop_last=True,
                            pin_memory=True,
                        )
    valid_dataloader = DataLoader(
                            args.valid_dataset,
                            batch_size=args.val_batch_size,
                            num_workers=1,
                            shuffle=False,
                            sampler = SequentialSampler(args.valid_dataset),
                            collate_fn=args.data_collator,
                            drop_last=True,
                            pin_memory=True,
                        )

    args.wav_config.pooling_mode = args.pooling_mode
    args.rnn_config.num_classes = args.num_classes

    if args.use_concat:
        model = MultiModel(args)
        model.wav_freeze_feature_extractor()
        model = model.to(args.device)
        
    else:
        if args.use_wav:
            wav_model = Wav2Vec2ForSpeechClassification.from_pretrained(args.wav_model, config=args.wav_config)
            wav_model.freeze_feature_extractor()
            model = wav_model.to(args.device)
        else:
            rnn_model = xlm_RoBertaBase.from_pretrained(args.rnn_model, config=args.rnn_config)
            model = rnn_model.to(args.device)


    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    total_valid_loss = np.inf
    total_valid_f1 = -np.inf

    # AMP
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        train_preds = []
        train_labels = []
        train_loss = 0.0
        bar = tqdm(train_dataloader)
        
        for idx, batch in enumerate(bar):
            optimizer.zero_grad()
            
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    pred, _ = sharing_step(args, model, batch)
                
                    loss = criterion(pred, batch['labels'])
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                pred, _ = sharing_step(args, model, batch)
                
                loss = criterion(pred, batch['labels'])
                loss.backward()
                optimizer.step()

            train_preds += [pred.clone().detach().cpu()]
            train_labels += [batch['labels'].clone().detach().cpu()]

            train_loss += loss.item()
            bar.set_description('loss : % 5f' % (loss.item()))

        train_loss /= idx
        train_preds, train_labels = np.concatenate(train_preds), np.concatenate(train_labels).reshape(-1)
        train_f1 = f1_score(train_labels, np.argmax(train_preds, 1), average='weighted')

        valids = valid_fn(args, valid_dataloader, model, criterion)
        valid_preds = valids['valid_preds']
        valid_loss = valids['val_loss']
        valid_f1 = valids['val_f1']

        # print train log & model save
        content = f'EPOCH : {epoch+1}, train_loss : {train_loss:.5f}, valid_loss : {valid_loss:.5f}, train_f1 : {train_f1:.5f}, valid_f1 : {valid_f1:.5f}'
        print(content)
        print(pd.Series(valid_preds).value_counts())
        
        # save checkpoint & logger
        if total_valid_f1<valid_f1:
            os.makedirs(args.output_dir + args.ver, exist_ok=True)
            
            with open(f'{args.output_dir}{args.ver}/log.txt', 'a') as appender:
                appender.write(content + '\n')
                appender.write('valid_preds value_counts' + '\n')
                appender.write(str(pd.Series(valid_preds).value_counts()) + '\n')
                appender.write(f'model performance F1 : {total_valid_f1:.4f} -> {valid_f1:.4f}' + '\n')
                appender.write('#################################################################' + '\n')
                
            if epoch>0:
                os.remove(model_path)
                
            print(f'model performance F1 : {total_valid_f1:.4f} -> {valid_f1:.4f}')
            total_valid_f1 = valid_f1
            
            model_path = content.replace(' ', '').replace(':', '').replace(',', '_')
            model_path = args.output_dir + args.ver + f'/{model_path}.pth'
            torch.save(model.state_dict(), model_path)
            
            if args.multimodal_method=='rsa':
                np.save(f'{args.output_dir}{args.ver}/attn_weights.npy', valids['attn_weights'])
        else:    
            with open(f'{args.output_dir}{args.ver}/log.txt', 'a') as appender:
                appender.write(content + '\n')
                appender.write('valid_preds value_counts' + '\n')
                appender.write(str(pd.Series(valid_preds).value_counts()) + '\n')
                appender.write('#################################################################' + '\n')
            
# validaion func
def valid_fn(args, valid_dataloader, model, criterion):
    
    model.eval()
    bar = tqdm(valid_dataloader)
    valid_preds = []
    valid_labels = []
    valid_loss = 0.0
    attn_weights = []

    with torch.no_grad():
        for idx, batch in enumerate(bar):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    pred, attn_w = sharing_step(args, model, batch)
                    
                    loss = criterion(pred, batch['labels'])
            else:
                pred, attn_w = sharing_step(args, model, batch)
                
                loss = criterion(pred, batch['labels'])
            
            valid_preds += [pred.clone().detach().cpu()]
            valid_labels += [batch['labels'].clone().detach().cpu()]
            
            if args.multimodal_method=='rsa':
                attn_weights += [attn_w.clone().detach().cpu().numpy()]

            valid_loss += loss.item()
            bar.set_description('loss : %.5f' % (loss.item()))
        
        valid_loss /= idx

    valid_preds, valid_labels = np.concatenate(valid_preds), np.concatenate(valid_labels).reshape(-1)    
    valid_preds = np.argmax(valid_preds, 1)
    
    if args.multimodal_method=='rsa':
        attn_weights = np.mean(np.mean(attn_weights, axis=0), axis=0)
    
    f1 = f1_score(valid_labels, valid_preds, average='weighted')
    valids = {  
                'valid_preds':valid_preds,
                'val_loss':valid_loss,
                'val_f1':f1,
                'attn_weights':attn_weights,
            }
    return valids
