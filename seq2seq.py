import torch
import torch.nn as nn
import torch.optim as optim
import random
#Defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source = "my name is yogesh sharma"
target = "<sos> i am yogesh <eos>"

source_vec = torch.tensor([[i for i, word in enumerate(source.split())]])
target_vec = torch.tensor([[i for i, word in enumerate(target.split())]])

# print(source_vec.shape, target_vec.shape)

# Encoder Block 
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        input_seq = input_seq.unsqueeze(0)
        embedded = self.embedding(input_seq)
        output,hidden = self.lstm(embedded, hidden)     
        return output, hidden

# Decoder Block
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        input_seq = input_seq.unsqueeze(0)
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)  
        prediction = self.out(output)
        return prediction, hidden   


# Seq_to_Seq Model
class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder =  decoder
        self.device = device
    
    def forward(self, src, trg, teacher_force_ratio):
        enc_inp_size = len(src[0])
        hidden = (torch.zeros(1, self.encoder.hidden_size).to(self.device), torch.zeros(1, self.encoder.hidden_size).to(self.device))
        dec_inp_size = len(trg[0])
        dec_inp = trg[0][0]

        #tensor to store decoder outputs
        outputs = torch.zeros(1, dec_inp_size)

        for t in range(enc_inp_size):
            out, hidden = encoder(src[0][t], hidden)

        for t in range(1 , dec_inp_size):
            prediction, hidden = decoder(dec_inp, hidden)

            outputs[0][t] = prediction.argmax(1)

            teacher_force = random.random() < teacher_force_ratio

            top = prediction.argmax(1).squeeze(0)

            dec_inp =  trg[0][t] if teacher_force else top

        return outputs.to(torch.float)



#Checking the Encoder
input_size = len(source_vec[0])

hidden_size = 20
encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
decoder = Decoder(hidden_size=hidden_size, output_size=len(target_vec[0]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_to_seq = seq2seq(encoder=encoder, decoder=decoder, device=device)
outputs = seq_to_seq(source_vec, target_vec, 0.75)
target_vec = target_vec.to(torch.float)
loss = nn.CrossEntropyLoss()
print(target_vec.to(device).dtype)


