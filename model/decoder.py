import torch
from model.token_embeddings import TokenEmbeddings
from model.positional_encodings import PositionalEncodings
from model.decoder_layer import DecoderLayer

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, h, N, dropout):
        super().__init__()
        self.token_embeddings = TokenEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncodings(seq_length, d_model, dropout)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layers = torch.nn.ModuleList([DecoderLayer(h, d_model, dropout) for _ in range(N)])
    
    def forward(self, target_seq, mask_dec, output_enc):
        token_embed = self.token_embeddings(target_seq)
        pos_encod_output = self.positional_encoding(token_embed)
        x = self.dropout(pos_encod_output)
        for layer in self.layers:
            x = layer(x, mask_dec, output_enc)
        return x