import json
import spacy
from torch import nn
import torch
from collections import Counter

# load models


nlp = spacy.load("en_core_web_sm")

pad_ix = 1

class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        #put padding_idx so asking the embedding layer to ignore padding
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_ix)
        self.lstm = nn.LSTM(emb_dim, 
                           hid_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, output_dim)
        
    def forward(self, text, text_lengths):
        #text = [batch size, seq len]
        embedded = self.embedding(text)
        
        #++ pack sequence ++
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False, batch_first=True)
        
        #embedded = [batch size, seq len, embed dim]
        packed_output, (hn, cn) = self.lstm(packed_embedded)  #if no h0, all zeroes
        
        #++ unpack in case we need to use it ++
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        #output = [batch size, seq len, hidden dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        return self.fc(hn)
        
# use this instead of text_pipeline
def text2index(txt,vocab_dict):
    
    txt_to = nlp(txt)

    present = []
    for word in txt_to:
        
        if str(word) in vocab_dict:
            value = vocab_dict[str(word)]
        else:
            value = 0

        present.append(value)      

    return present

def load_lstm():
    save_path = "models/LSTM.pt"
    json_dir ="models/vocab.json"
    with open(json_dir, "r") as file:
        vocab_dict = json.load(file)

    input_dim  = len(vocab_dict)
    hid_dim    = 256
    emb_dim    = 300       
    output_dim = 2
    
    #for biLSTM
    num_layers = 2
    bidirectional = True
    dropout = 0.5

    model_load = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout)

    model_load.load_state_dict(torch.load(save_path))

    return model_load,vocab_dict


def top_three_words(word_list):
    word_count = Counter(word_list)
    top_three = word_count.most_common(3)
    return [word[0] for word in top_three]

def predict_text(test_str,model_load,vocab_dict):

    text = torch.tensor(text2index(test_str,vocab_dict)).reshape(1, -1) 
    text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)

    with torch.no_grad():
        output = model_load(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1]
        return predicted.item()

if __name__ == "__main__":


    model,vocab_dict = load_lstm()

    test_str = "Chaky wants his student to be number 1."
    if predict_text(test_str,model):
        print("positive")
    else:
        print("negative")




