for bs in [32, 64, 128, 256]:
    for lr in [0.001, 0.005, 0.0001, 0.0005]:
        print(f"{bs},{lr}")


    # for embed_dims in [16, 32, 64]:
    #     for latent_dims in [16, 32, 64]:
    #             for beta_weight in [1, 2]:
    #                 for encoder_dims in ["\"[128,128]\"", "\"[512,512]\"", "\"[1024,1024]\""]:
    #                     for enc_dropout in [0., 0.2, 0.5]:
    #                         for decoder_dims in ["\"[128,128]\"", "\"[512,512]\"", "\"[1024,1024]\""]:
    #                             for decoder_dropout in [0., 0.2, 0.5]:
    #                                 for prior_dims in ["\"[16,16]\"", "\"[32,32]\""]:
    #                                      print(f"{bs},{embed_dims},{latent_dims},{lr},{beta_weight},{encoder_dims},{enc_dropout},{decoder_dims},{decoder_dropout},{prior_dims}")


