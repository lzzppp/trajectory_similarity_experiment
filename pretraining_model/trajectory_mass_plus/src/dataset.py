from torch.utils.data import Dataset
import pandas as pd

class myDataset(Dataset):
    def __init__(self, b_d_tokens, b_d_times, b_d2_tokens, b_d2_times, b_d_positions, p_targets, l1s, l2s, p_t_masks):
        self.b_d_tokens = b_d_tokens
        self.b_d_times = b_d_times
        self.b_d2_tokens = b_d2_tokens
        self.b_d2_times = b_d2_times
        self.b_d_positions = b_d_positions
        self.p_targets = p_targets
        self.l1s = l1s
        self.l2s = l2s
        self.p_t_masks = p_t_masks

    def __len__(self):
        return len(self.b_d_tokens)

    def __gettime__(self,idx):
        data = (self.b_d_tokens[idx],
                self.b_d_times[idx],
                self.b_d2_tokens[idx],
                self.b_d2_times[idx],
                self.b_d_positions[idx],
                self.p_targets[idx],
                self.l1s[idx],
                self.l2s[idx],
                self.p_t_masks[idx])
        return data