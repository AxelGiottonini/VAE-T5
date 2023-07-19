import random
import torch
import torch.nn.functional as F

class Tokens():
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        return self
    
class MaskedTokens(Tokens):
    def __init__(self, masked_input_ids, masked_mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_input_ids = masked_input_ids
        self.masked_mask = masked_mask

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.masked_input_ids = self.masked_input_ids.to(*args, **kwargs)
        self.masked_mask = self.masked_mask.to(*args, **kwargs)
        return self

def __get_collate_fn__(tokenizer, mask, mask_rate, frag_coef_a=0, frag_coef_b=1, split=True):
    def collate_fn(seqs):
        if not frag_coef_a == 0 and not frag_coef_b == 1:
            seqs = [el[(start:=(random.randint(0, len(el)//frag_coef_a))):(stop:=(random.randint(start+len(el)//frag_coef_b, len(el))))] for el in seqs]
        
        if split:
            seqs = [' '.join(list(el)) for el in seqs]

        tokens = tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask

        if mask:
            masked_mask = F.dropout(attention_mask.to(torch.float32), (1-mask_rate))
            masked_input_ids = ((1-masked_mask)*input_ids + masked_mask*tokenizer.mask_token_id).to(torch.long)

            tokens = MaskedTokens(
                input_ids = input_ids,
                masked_input_ids = masked_input_ids,
                attention_mask = attention_mask,
                masked_mask = masked_mask
            )
            return tokens

        tokens = Tokens(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        return tokens

    return collate_fn