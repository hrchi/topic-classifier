import torch
from torch.nn.utils.rnn import pad_sequence


def transformer_collate_fn(pad_idx):
    def collate(batch):
        input_ids, labels = zip(*batch)
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)  
        src_key_padding_mask = (padded_input_ids == pad_idx)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded_input_ids, labels, src_key_padding_mask
    return collate