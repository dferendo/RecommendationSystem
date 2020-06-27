for type in ['linear', 'sigmoid', 'cosine']:
    for max_beta in [1, 2, 5]:
        for cycles in [1, 2, 4]:
            for ratio in [0.25]:
                print(f"{type},{max_beta},{cycles},{ratio}")


    # for embed_dims in [16, 32, 64]:
    #     for latent_dims in [16, 32, 64]:
    #             for beta_weight in [1, 2]:
    #                 for encoder_dims in ["\"[128,128]\"", "\"[512,512]\"", "\"[1024,1024]\""]:
    #                     for enc_dropout in [0., 0.2, 0.5]:
    #                         for decoder_dims in ["\"[128,128]\"", "\"[512,512]\"", "\"[1024,1024]\""]:
    #                             for decoder_dropout in [0., 0.2, 0.5]:
    #                                 for prior_dims in ["\"[16,16]\"", "\"[32,32]\""]:
    #                                      print(f"{bs},{embed_dims},{latent_dims},{lr},{beta_weight},{encoder_dims},{enc_dropout},{decoder_dims},{decoder_dropout},{prior_dims}")


