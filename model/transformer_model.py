import torch
from model.encoder import Encoder
from model.decoder import Decoder

class TransformerModel(torch.nn.Module):
    def __init__(self, enc_type, nb_samples, max_nb_var, d_model, vocab_size, seq_length, h, N_enc, N_dec, dropout):
        super().__init__()
        self.encoder = Encoder(enc_type, nb_samples, max_nb_var, d_model, h, N_enc, dropout)
        self.decoder = Decoder(vocab_size, seq_length, d_model, h, N_dec, dropout)
        self.last_layer = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, input_enc, target_seq):
        padding_mask = torch.eq(target_seq, 0).unsqueeze(1).unsqueeze(1)
        future_mask = torch.triu(torch.ones(target_seq.shape[1], target_seq.shape[1]), diagonal=1).bool()
        mask_dec = torch.logical_or(padding_mask, future_mask)
        encoder_output = self.encoder(input_enc)
        decoder_output = self.decoder(target_seq, mask_dec, encoder_output)
        final_output = self.last_layer(decoder_output)
        return final_output