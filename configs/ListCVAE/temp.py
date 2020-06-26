for enc_1 in ["true", "false"]:
    for enc_2 in [0.1, 0.3]:
        for enc_3 in ["leaky"]:
            for dec_1 in ["true", "false"]:
                for dec_2 in [0.1, 0.3]:
                    for dec_3 in ["leaky"]:
                        print(f"{enc_1},{enc_2},{enc_3},{dec_1},{dec_2},{dec_3}")