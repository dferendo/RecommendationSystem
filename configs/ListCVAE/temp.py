# # for type in ['linear', 'sigmoid', 'cosine']:
# #     for max_beta in [1, 2, 5]:
# #         for cycles in [1, 2, 4]:
# #             for ratio in [0.25]:
# #                 print(f"{type},{max_beta},{cycles},{ratio}")
#
# # train_batch_size,lr,embed_dims,latent_dims,encoder_dims,enc_dropout,decoder_dims,dec_dropout,
# #
# # batch_norm,activation
#
# for batch_size in [32, 128, 256]:
#     for lr in [0.001, 0.0001]:
#         for architecture in ["16,16,\"[128,128]\",0.0,\"[128,128]\",0.0"]:
#             print(f"{batch_size},{lr},{architecture}")
#
#
#
#             #     for decoder_dims in ["\"[128,128]\"", "\"[512,512]\"", "\"[1024,1024]\""]:
#             #             # for decoder_dropout in [0., 0.2, 0.5]:
#             #         for prior_dims in ["\"[16,16]\"", "\"[32,32]\""]:
#             #             print(f"{embed_dims},{latent_dims},{encoder_dims},{decoder_dims},{prior_dims}")
#
#

for a in [0.001, 0.005, 0.0001, 0.0005]:
    for b in [8, 16, 32, 64, 128]:
        for c in [0, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
            print(f"{a},{b},{c}")
