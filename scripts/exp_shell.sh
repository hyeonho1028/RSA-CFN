###################### Exp Uni Model (Wav)
python main.py --ver wav_model \
               --use_wav \
               --label_transform

###################### Exp Uni Model (Text)
python main.py --ver text_model \
               --label_transform 

###################### Exp MLP_Mixer
python main.py --ver mlp_mixer \
               --use_concat --multimodal_method mlp_mixer \
               --label_transform 

###################### Exp Base Early Fusion
python main.py --ver early_fusion \
               --use_concat --multimodal_method early_fusion \
               --label_transform

###################### Exp Late Fusion
python main.py --ver late_fusion \
               --use_concat --multimodal_method late_fusion \
               --label_transform  

###################### Exp Stack arch. Early Fusion
python main.py --ver stacking \
               --use_concat --multimodal_method stack \
               --label_transform  

###################### Exp Residual arch. Early Fusion
python main.py --ver residual \
               --use_concat --multimodal_method residual \
               --label_transform  

###################### Exp Residual with Self Attention arch. Early Fusion
python main.py --ver rsa \
               --use_concat --multimodal_method rsa \
               --label_transform  

###################### Exp residuals attention cross fusion
python main.py --ver rsa_cfn \
               --use_concat --multimodal_method rsa_cfn \
               --label_transform  

###################### Exp Hybrid Fusion
python main.py --ver hybrid_fusion \
               --use_concat --multimodal_method hybrid_fusion \
               --label_transform 
               