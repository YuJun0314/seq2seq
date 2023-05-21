import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def get_data():
    english = ["apple", "banana", "orange", "pear", "black", "red", "white", "pink", "green", "blue"]
    chinese = ["苹果", "香蕉", "橙子", "梨", "黑色", "红色", "白色", "粉红色", "绿色", "蓝色"]
    return english, chinese

def get_word_dict(english, chinese):
    eng_to_idx = {"PAD":0, "UNK":1}
    chn_to_idx = {"PAD":0, "UNK":1,"STA":2 ,"END":3}

    for eng in english:
        for w in eng:
            eng_to_idx[w] = eng_to_idx.get(w, len(eng_to_idx))

    for chn in chinese:
        for w in chn:
            chn_to_idx[w] = chn_to_idx.get(w, len(chn_to_idx))

    return eng_to_idx, list(eng_to_idx), chn_to_idx, list(chn_to_idx)

class TDataset(Dataset):
    def __init__(self,english,chinese,param):
        self.param = param
        self.english = english
        self.chinese = chinese

    def __getitem__(self, index):
        e_data = self.english[index][:self.param["eng_max_len"]]
        c_data = self.chinese[index][:self.param["chn_max_len"]]
        e_index = [self.param["eng_to_idx"].get(i, 1) for i in e_data] + [0]*(self.param["eng_max_len"]-len(e_data))
        c_index = [2] + [self.param["chn_to_idx"].get(i, 1) for i in c_data] +[3]+ [0]*(self.param["chn_max_len"]-len(c_data))

        return torch.tensor(e_index), torch.tensor(c_index)

    def __len__(self):
        return len(self.english)

class Encoder(nn.Module):
    def __init__(self,param):
        super().__init__()
        self.embedding = nn.Embedding(len(param["eng_to_idx"]), param["embedding_num"])
        self.backbone = nn.GRU(param["embedding_num"], param["hidden_num"], batch_first=True, bidirectional=param["bi"])
    def forward(self,batch_index):
        emb = self.embedding.forward(batch_index)
        _, hidden = self.backbone.forward(emb)
        return hidden

class Decoder(nn.Module):
    def __init__(self,param):
        super().__init__()
        self.embedding = nn.Embedding(len(param["eng_to_idx"]), param["embedding_num"])
        self.backbone = nn.GRU(param["embedding_num"], param["hidden_num"], batch_first=True, bidirectional=param["bi"])
        self.att_linear = nn.Linear(param["hidden_num"], 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,batch_index, hidden):
        emb = self.embedding.forward(batch_index)
        out, hidden = self.backbone(emb, hidden)
        att_out = self.att_linear(out)
        score = self.softmax(att_out)
        out = score * out

        return out, hidden

class TModel(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.Encoder = Encoder(param)
        self.Decoder = Decoder(param)
        self.classifier = nn.Linear(param["hidden_num"], len(param["chn_to_idx"]))
        self.loss_fun = nn.CrossEntropyLoss()


    def forward(self, eng_index, chn_index=None):
        encoder_hidden = self.Encoder.forward(eng_index)
        out, hidden = self.Decoder.forward(chn_index[:,:-1], encoder_hidden)
        pre = self.classifier.forward(out)
        loss = self.loss_fun(pre.reshape(-1,pre.shape[-1]), chn_index[:,1:].reshape(-1))

        return loss

    def trainslate(self, eng_index,index_to_chn):
        assert len(eng_index) == 1
        result = []

        eng_e = self.eng_embedding(eng_index)
        _, encoder_out = self.encoder(eng_e)

        decoder_hid = encoder_out

        chn_index = torch.tensor([[2]])

        while True:
            chn_e = self.chn_embedding.forward(chn_index)
            decoder_out,decoder_hid = self.decoder.forward(chn_e,decoder_hid)
            pre = self.classifier.forward(decoder_out)
            chn_index = torch.argmax(pre, dim=-1)


            if int(chn_index) == 3 or len(result)>20:
                break
            word = index_to_chn[int(chn_index)]
            result.append(word)
        return "".join(result)

if __name__ == "__main__":
    english, chinese = get_data()
    eng_to_idx, idx_to_eng, chn_to_idx, idx_to_chn = get_word_dict(english, chinese)
    param = {
        "eng_to_idx":eng_to_idx,
        "idx_to_eng":idx_to_eng,
        "chn_to_idx":chn_to_idx,
        "idx_to_chn":idx_to_chn,
        "hidden_num":10,
        "embedding_num":20,
        "chn_max_len":3,
        "eng_max_len":6,
        "batch_size":2,
        "epoch":100,
        "bi":False,
        "lr":1e-3
    }
    dataset = TDataset(english, chinese, param)
    dataloader = DataLoader(dataset, batch_size=param["batch_size"],shuffle=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = TModel(param).to(device)
    opt = torch.optim.AdamW(model.parameters(), param["lr"])

    epoch_total_loss = 0
    best_loss = 9999

    for e in range(param["epoch"]):
        for eng_index, chn_index in tqdm(dataloader,desc="training..."):
            eng_index = eng_index.to(device)
            chn_index = chn_index.to(device)
            loss = model.forward(eng_index, chn_index)
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_total_loss += loss

        if best_loss > epoch_total_loss:
            print("save best model")
            torch.save(model.state_dict(),"best_model.pt")
            best_loss = epoch_total_loss
        print(f"epoch_total_loss:{epoch_total_loss:.3f} ,   best_loss:{best_loss:.3f}\n")
        epoch_total_loss = 0

    with open("param.pkl","wb") as f:   # w写b二进制
        pickle.dump(param, f)