import json
import csv
import os

data_dir = 'data/goldpaths-all'
raw_data_list = []

for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), 'r') as f:
        raw_data_list.append(json.load(f))

data = []
train_data = []
val_data = []
test_data = []

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

for raw_data in raw_data_list:
    for task_id in raw_data.keys():
        curr_task = raw_data[task_id]
        for seq_sample in curr_task['goldActionSequences']:
            task_desc = seq_sample['taskDescription']
            steps = seq_sample['path']
            if len(steps) < 2:
                continue
            fold = seq_sample['fold']
            obs = steps[0]['observation']
            action = steps[0]['action']
            for i in range(len(steps)):
                if i != 0:
                    prev_step = steps[i-1]
                    curr_step = steps[i]

                    prev_action = prev_step['action']
                    curr_action = curr_step['action']
                    prev_obs = prev_step['observation']
                    curr_obs = curr_step['observation']
                    look = curr_step['freelook']
                    inventory = curr_step['inventory']

                    if curr_action == 'look around':
                        continue

                    input_str = '[CLS] ' + task_desc + ' [SEP] ' + curr_obs + ' [SEP] ' + inventory + ' [SEP] ' + look + ' [SEP] '\
                        + prev_obs + ' [SEP] ' + prev_action + ' [SEP]'
                    target = f"{curr_action} [SEP]"


                    
                else:
                    curr_step = steps[i]
                    curr_action = curr_step['action']
                    curr_obs = curr_step['observation']
                    look = curr_step['freelook']
                    inventory = curr_step['inventory']

                    if curr_action == 'look around':
                        continue

                    input_str = '[CLS] ' + task_desc + ' [SEP] ' + curr_obs + ' [SEP] ' + inventory + ' [SEP] ' + look + ' [SEP] '\
                        + '[SEP] [SEP]'
                    target = f"{curr_action} [SEP]"

                curr_dat = {'input': clean(input_str), 'target':clean(target)}

                if fold == 'train':
                    train_data.append(curr_dat)
                elif fold == 'dev':
                    val_data.append(curr_dat)
                elif fold == 'test':
                    test_data.append(curr_dat)

mlen = len(data)

if val_data == []:
    val_data = data[int(0.9 * mlen):]
if test_data == []:
    test_data = data[int(0.1 * mlen):int(0.2 * mlen)]

with open("sciworld_formatted_train.jsonl", 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("sciworld_formatted_val.jsonl", 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open("sciworld_formatted_test.jsonl", 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")
