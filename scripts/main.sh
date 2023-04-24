# ###################### Exp Base Early Fusion
# python main.py --ver base \
#                --use_concat --multimodal_method base \
#                --label_transform \


# ###################### Exp residuals attention cross fusion
# python main.py --ver late_fusion \
#                --use_concat --multimodal_method late_fusion \
#                --label_transform \


# ###################### Exp Stack arch. Early Fusion
# python main.py --ver stacking \
#                --use_concat --multimodal_method stacking \
#                --label_transform \


# ###################### Exp Residuals arch. Early Fusion
# python main.py --ver residuals \
#                --use_concat --multimodal_method residuals \
#                --label_transform \


# ###################### Exp Residuals with Self Attention arch. Early Fusion
# python main.py --ver residuals_attn \
#                --use_concat --multimodal_method residuals_attn \
#                --label_transform \


###################### Exp residuals attention cross fusion
python main.py --ver rsa_cfn_origin_gelu \
               --use_concat --multimodal_method rsa_cfn \
               --label_transform 

# ###################### Exp Hybrid Fusion
# python main.py --ver hybrid_fusion \
#                --use_concat --multimodal_method hybrid_fusion \
#                --label_transform

