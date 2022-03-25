import torch
import os
import json
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def get_dataset(data_file, tokenizer, max_len=-1, shuffle_trajectories=False, data_percentage=1):
    token_id_set, act_mask_set = [], []

    with open(os.path.join(data_file), 'r') as f:
        data_list = list(f)
        if shuffle_trajectories:
            data_list = random.shuffle(data_list)
        data_list = data_list[:int(len(data_list)*data_percentage)]
        for json_str in data_list:
            data = json.loads(json_str)
            observation_ids = tokenizer.encode(data["input"])
            action_ids = tokenizer.encode(data["target"])
            observation_len = len(observation_ids)
            action_len = len(action_ids)
            token_ids = observation_ids + action_ids
            act_mask = [0] * observation_len + [1] * action_len
            if max_len == -1 or len(token_ids) < max_len:
                token_id_set.append(token_ids)
                act_mask_set.append(act_mask)


    return token_id_set, act_mask_set


def process(line, tokenizer):
    """
    Process each line of the dataset to tokens and action masks.
    :param act_len: Pad or cut action length to act_len. 7 for BERT model, 1 for verb model, -1 for doing nothing.
    if -2 this means we mask out the last sep token for gpt-2
    """
    # Turn [STATE] and [ACTION] to [SEP]
    words = line.split()
    while "" in words:
        words.remove("")
    words = ["[SEP]" if w in ["[STATE]", "[ACTION]"] else w for w in words]
    words[0] = "[CLS]"
    line = " ".join(words)

    # Find where the last action starts, and cut or pad when needed
    tokens = tokenizer.tokenize(line, add_prefix_space=True)
    act_pos = len(tokens) - tokens[::-1].index("[SEP]")
    tokens += ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Act mask
    act_mask = np.zeros(len(tokens))
    act_mask[act_pos:] = 1

    return token_ids, act_mask

def pad_sequences(data, pad_length, dtype):
    padded_data = np.zeros((len(data), pad_length), dtype=dtype)
    for i, line in enumerate(data):
        if len(line) > pad_length:
            line = line[len(line) - pad_length:]
        padded_data[i,:len(line)] = line
    return padded_data

def train_test_split(data, validate_size=0.9):
    token_ids, act_masks, att_masks = data
    indices = [i for i in range(len(token_ids))]
    train_idx = indices[:int(validate_size * len(indices))]
    validate_idx = indices[int(validate_size * len(indices)):]

    train_inputs = token_ids[train_idx]
    val_inputs = token_ids[validate_idx]
    train_act_masks = act_masks[train_idx]
    val_act_masks = act_masks[validate_idx]
    train_att_masks = att_masks[train_idx]
    val_att_masks = att_masks[validate_idx]

    return train_inputs, val_inputs, train_act_masks, val_act_masks, train_att_masks, val_att_masks


def _get_dataloader(data_directory, tokenizer, max_len=256, bs=16, shuffle_trajectories=False,
                   data_percentage=1):
    n_gpu = torch.cuda.device_count()
    per_gpu_batch_size = bs
    batch_size = max(1, n_gpu) * per_gpu_batch_size
    print("Number of GPU: " + str(n_gpu))

    token_id_set, act_mask_set = get_dataset(data_directory, tokenizer, max_len=max_len, \
                                             shuffle_trajectories=shuffle_trajectories, data_percentage=data_percentage)
    att_mask_set = [np.ones(len(ids)) for ids in token_id_set]
    print(str(len(token_id_set)) + " examples in dataset")
    print("Data Sample\n", tokenizer.convert_ids_to_tokens(token_id_set[0]), '\n', act_mask_set[0])

    token_ids = pad_sequences(token_id_set, 512, 'int')
    act_masks = pad_sequences(act_mask_set, 512, 'uint8')
    att_masks = pad_sequences(att_mask_set, 512, 'uint8')

    data = TensorDataset(torch.tensor(token_ids), torch.tensor(act_masks), torch.tensor(att_masks))
    data_sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=data_sampler, batch_size=batch_size,
                                  drop_last=True)  # drop last batch for gpt-2

    return dataloader

def get_dataloader(train_data, val_data, tokenizer, max_len=256, bs=16, shuffle_trajectories=False,
                   data_percentage=1):
    train_dataloader = _get_dataloader(train_data, tokenizer, max_len=max_len, bs=bs, shuffle_trajectories=shuffle_trajectories,
                   data_percentage=data_percentage)
    val_dataloader = _get_dataloader(val_data, tokenizer, max_len=max_len, bs=bs, shuffle_trajectories=shuffle_trajectories,
                   data_percentage=data_percentage)
    return train_dataloader, val_dataloader