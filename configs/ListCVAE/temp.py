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

import torch

A = torch.tensor([[1.0124e+08, 9.9517e+07, 5.3546e+07, 7.5420e+07, 1.0620e+08, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [9.9517e+07, 1.0104e+08, 5.5640e+07, 7.5404e+07, 1.0777e+08, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [5.3546e+07, 5.5640e+07, 4.2558e+07, 4.2080e+07, 6.3944e+07, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [7.5420e+07, 7.5404e+07, 4.2080e+07, 5.8183e+07, 8.2586e+07, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [1.0620e+08, 1.0777e+08, 6.3944e+07, 8.2586e+07, 1.1964e+08, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0124e+08,
         9.9517e+07, 5.3546e+07, 7.5420e+07, 1.0620e+08],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.9517e+07,
         1.0104e+08, 5.5640e+07, 7.5404e+07, 1.0777e+08],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.3546e+07,
         5.5640e+07, 4.2558e+07, 4.2080e+07, 6.3944e+07],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.5420e+07,
         7.5404e+07, 4.2080e+07, 5.8183e+07, 8.2586e+07],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0620e+08,
         1.0777e+08, 6.3944e+07, 8.2586e+07, 1.1964e+08]],requires_grad=True)
# A1 = A.detach()
# eigvalng, eigvecng = torch.symeig(A1,eigenvectors=True)
# eps_matrix = torch.diag(torch.randn(10)*1e8)
# eps_matrix = eigvecng.mm(eps_matrix)
# eps_matrix = eps_matrix.mm(torch.transpose(eigvecng,0,1))

temp = torch.randn(A.size()) * 1e8
eigval, eigvec = torch.symeig(A, eigenvectors=True)
maxid = torch.argmax(eigval)
bvector = eigvec[:,maxid]
loss = torch.sum(bvector)
loss.backward()
# print (loss)
# print (A.grad)
print(temp)
