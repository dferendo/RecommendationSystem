# for type in ['linear', 'sigmoid', 'cosine']:
#     for max_beta in [1, 2, 5]:
#         for cycles in [1, 2, 4]:
#             for ratio in [0.25]:
#                 print(f"{type},{max_beta},{cycles},{ratio}")


# for embed_dims in [16, 32, 64]:
#     for latent_dims in [16, 32, 64]:
#         print(f"{embed_dims},{latent_dims}")
for encoder_dims in ["\"[128,128]\"", "\"[256,256]\"", "\"[512,512]\"", "\"[1024,1024]\"", "\"[512,512,512]\"", "\"[1024,1024,1024]\""]:
    for enc_dropout in [0., 0.2, 0.5]:
        print(f"{encoder_dims},{enc_dropout}")


            #     for decoder_dims in ["\"[128,128]\"", "\"[512,512]\"", "\"[1024,1024]\""]:
            #             # for decoder_dropout in [0., 0.2, 0.5]:
            #         for prior_dims in ["\"[16,16]\"", "\"[32,32]\""]:
            #             print(f"{embed_dims},{latent_dims},{encoder_dims},{decoder_dims},{prior_dims}")


