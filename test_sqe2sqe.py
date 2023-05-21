import torch
import torch.nn as nn
import pickle
from seq2seq import TModel

# class TModel(nn.Module):
#     def __init__(self, param):
#         super().__init__()
#         self.eng_embedding = nn.Embedding(len(param["idx_to_eng"]), param["embedding_num"])
#         self.chn_embedding = nn.Embedding(len(param["idx_to_chn"]), param["embedding_num"])
#         self.encoder = nn.GRU(param["embedding_num"],param["hidden_num"],batch_first=True, bidirectional=param["bi"])
#         self.decoder = nn.GRU(param["embedding_num"],param["hidden_num"],batch_first=True, bidirectional=param["bi"])
#         self.classifier = nn.Linear(param["hidden_num"]*2,len(param["idx_to_chn"]))
#         self.loss_fun = nn.CrossEntropyLoss()
#
#     def forward(self, eng_index, chn_index):
#         eng_e = self.eng_embedding(eng_index)
#         chn_e = self.chn_embedding(chn_index)
#         m,encoder_out = self.encoder.forward(eng_e)
#         decoder_out,_ = self.decoder.forward(chn_e,encoder_out)
#         pre = self.classifier.forward(decoder_out)
#         loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), chn_index.reshape(-1))
#         return loss

if __name__ == "__main__":
    with open("param.pkl", "rb") as f:
        param = pickle.load(f)

    model = TModel(param)
    model_param = torch.load("best_model.pt", map_location="cpu")
    model.load_state_dict(model_param)

    while True:
        #input_text = input("请输入：")
        input_text = "banana"
        input_idx = [param["eng_to_idx"].get(i,1) for i in input_text]
        input_idx = torch.tensor([input_idx])

        result = model.trainslate(input_idx,param["idx_to_chn"])
        print(result)