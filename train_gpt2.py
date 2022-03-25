import os
import json
import torch
import argparse
from transformers import WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_for_scienceworld_recording import *
from lm import *

def train(train_dataloader, validation_dataloader, lm, save_dir_root, args):
    return _train_gpt(train_dataloader, validation_dataloader, lm, save_dir_root, args)

def _train_gpt(train_dataloader, validation_dataloader, lm, save_dir_root, args):
    model = lm.model
    tokenizer = lm.tokenizer
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    adam_epsilon = 1e-8
    warmup_steps = .1
    weight_decay = 0
    max_grad_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    t_total = len(train_dataloader) // gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    global_step = 0
    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = range(0, int(args.num_train_epochs))
    train_losses, val_losses = [], []

    for _ in train_iterator:
        print("Beginning Epoch: " + str(_))
        epoch_iterator = tqdm(train_dataloader)
        total_actions = 0
        tr_loss = 0
        for step, batch in enumerate(epoch_iterator):
            b_input_ids, b_input_mask, b_strat = batch
            b_labels = b_input_ids.clone()
            b_labels[b_strat == 0] = -100
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = b_input_mask.to(device)

            model.train()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            loss_value = loss.item()
            tr_loss += loss_value
            total_actions += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        train_losses.append(tr_loss / total_actions)
        val_loss= validate(validation_dataloader, lm)
        val_losses.append(val_loss)
        print('epoch %02d: train loss %.2lf, val loss %.2lf\n' % (
            _,  train_losses[-1], val_losses[-1]))

        if _ % args.freq_save_epochs == 0:
            save_gpt(model, tokenizer, save_dir_root, '%s/epoch%02d' % (args.model_name, _))

    print("Total Iterations: " + str(global_step))
    return train_losses, val_losses

def validate(eval_dataloader, lm):
    return _validate_gpt(eval_dataloader, lm)


def _validate_gpt(eval_dataloader, lm):
    model = lm.model
    eval_loss = 0.0
    nb_eval_steps = 0
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(eval_dataloader):
        b_input_ids, b_input_mask, b_strat = batch
        b_labels = b_input_ids.clone()
        b_labels[b_strat == 0] = -100

        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)
        b_input_mask = b_input_mask.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        lm_loss = outputs[0]
        eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps


    return eval_loss

def save_gpt(model, tokenizer, save_dir_root, name):
    output_dir = os.path.join(save_dir_root, name)
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', default='model')
    parser.add_argument('--model_path', default='gpt2')
    parser.add_argument('--initialization', default='pretrained', help='pretrained | random')
    parser.add_argument('--save_dir_root', default='calm/finetune/models', type=str)

    # data
    parser.add_argument('--bs', default=1, type=int, help='batch size per gpu')
    parser.add_argument('--shuffle_trajectories', default=0, type=int)
    parser.add_argument('--train_data', default='sciworld_formatted_train.jsonl', type=str)
    parser.add_argument('--val_data', default = "sciworld_formatted_val.jsonl", type=str)
    parser.add_argument('--max_len', default=500, type=int, help='max #tokens allowed in one line')
    parser.add_argument('--data_percentage', default=1, type=float, help='percentage of games to use')

    # training
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument('--freq_save_epochs', default=1, type=int)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    model = GPT2LM(args.model_path)


    train_data, validation_data = get_dataloader(args.train_data, args.val_data, model.tokenizer,
                                                    max_len=args.max_len,
                                                    shuffle_trajectories=args.shuffle_trajectories == 1,
                                                    bs=args.bs, data_percentage=args.data_percentage)

    stats = train(train_data, validation_data, model, os.path.abspath(args.save_dir_root), args)

    stats = list(zip(*stats))

    json.dump(stats, open(os.path.join(args.save_dir_root, args.model_name, 'stats.json'), 'w'), indent=4)
