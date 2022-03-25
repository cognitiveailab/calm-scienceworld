import os
import time
from drrn import build_state
from memory import State
import argparse
import logging
import json
from statistics import mean
from env import CALMScienceWorldEnv
from random import choice
from collections import defaultdict
import torch
import random

import sys
from scienceworld import BufferedHistorySaver

from lm import *
from drrn import *


class CALMModel(object):
    def __init__(self, args) -> None:
        self.args = args
        self.lm = GPT2LM(args.lm_path, args.tokenizer_path)
        self.lm.model = self.lm.model.eval()
        self.args.vocab_size = self.lm.tokenizer.vocab_size
        print(self.args)

        self.agent = DRRN_Agent(args)
        self.init_envs()

        self.train_scores = []
        self.eval_scores = []

    def init_envs(self):
        args = self.args

        env = CALMScienceWorldEnv(args.jar_path,args.seed, args.task_num, args.env_step_limit, args.stuck_step_limit, get_valid=args.lm_top_k == 0, 
                                simplification_str=args.simplification_str, thread_offset=0)
        env.create(99, 0)

        self.train_var_nos = list(env.getVariationsTrain())
        self.dev_var_nos = list(env.getVariationsDev())
        self.test_var_nos = list(env.getVariationsTest())
        if self.args.output_dir.endswith('/'):
            self.args.output_dir = self.args.output_dir[:-1]
        self.bufferedHistorySaverTrain = BufferedHistorySaver(filenameOutPrefix = f"{self.args.output_dir}/calm-saveout" + "-seed" + str(args.seed) + "-task" + str(self.args.task_num) + "-train")
        self.bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix = f"{self.args.output_dir}/calm-saveout" + "-seed" + str(args.seed) + "-task" + str(self.args.task_num) + "-eval")
        env.close()

    def evaluate(self, train_steps):
        step = 0
        done = False
        
        start = time.time()
        obs, rewards, dones, infos, transitions = [], [], [], [], []
        env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)

        eval_envs = [CALMScienceWorldEnv(self.args.jar_path, self.args.seed, self.args.task_num, self.args.env_step_limit, self.args.stuck_step_limit,
                                     get_valid=self.args.lm_top_k == 0, simplification_str=self.args.simplification_str, thread_offset=100)
                                     for _ in range(self.args.num_envs)]

        if self.args.lm_dict:
            d_in = defaultdict(list, json.load(open('%s/d_in.json' % self.args.lm_dict, 'r')))
            d_out = defaultdict(list, json.load(open('%s/d_out.json' % self.args.lm_dict, 'r')))
            self.lm.generate_dict = json.load(open('%s/lm.json' % self.args.lm_dict, 'r'))
        for n, env in enumerate(eval_envs):
            env.create(n, 0)
            variation = random.choice(self.test_var_nos)
            env.load(variation)
            ob, info = env.reset()
            obs, rewards, dones, infos, transitions = \
                obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]]
        states = build_state(self.lm, obs, infos)
        valid_ids = [[self.lm.act2ids(a) for a in info['valid']] for info in infos]

        episode = 0
        while episode < self.args.max_eval_episodes:
            # act
            action_ids, action_idxs, action_values = self.agent.act(states, valid_ids, lm=self.lm,
                                                            eps=self.args.eps, alpha=self.args.alpha, k=self.args.eps_top_k)
            action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

            # step with rejection
            next_obs, next_rewards, next_dones, next_infos = [], [], [], []
            for i, (env, action) in enumerate(zip(eval_envs, action_strs)):
                if dones[i]:
                    if env.max_score >= max_score:  # put in alpha queue
                        for transition in transitions[i]:
                            self.agent.observe(transition, is_prior=True)
                    episode += 1
                    
                    self.eval_scores.append(infos[i]['score'])
                    if episode < 9:
                        last_10_score = mean(self.eval_scores)
                    else:
                        last_10_score = mean(self.eval_scores[int(len(self.eval_scores)*0.9):])
                    run_history = infos[i]['history']
                    self.bufferedHistorySaverEval.storeRunHistory(run_history, f"{train_steps}-{episode}", notes={'last_10_score':last_10_score})
                    self.bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=self.args.max_histories_per_file)
                    if episode >= self.args.max_eval_episodes:
                        return
                    variation = random.choice(self.test_var_nos)
                    env.load(variation)
                    ob, info = env.reset()
                    action_strs[i], obs[i], action_ids[i], transitions[i] = '', '', [], []
                    next_obs, next_rewards, next_dones, next_infos = next_obs + [ob], next_rewards + [0], next_dones + [
                        False], next_infos + [info]
                    continue
                prev_inv, prev_look = infos[i]['inv'], infos[i]['look']
                ob, reward, done, info = env.step(action)
                if self.args.lm_top_k:  # deal with rejection
                    key = hash(tuple(states[i][0] + states[i][1] + states[i][2]))
                    l_in, l_out = d_in[key], d_out[key]
                    actions = infos[i]['valid']
                    rej = 'Input: No known action matches that input.' in ob

                    # while action is invalid, pull another action from CALM generated candidates
                    while not done and rej and len(actions) > 1:
                        if action not in l_out: l_out.append(action)
                        actions.remove(action)
                        action = choice(actions)
                        ob, reward, done, info = env.step(action)
                        rej = 'Input: No known action matches that input.' in ob
                    action_strs[i] = action

                    if not rej and action not in l_in: l_in.append(action)
                    if reward < 0 and action not in l_out: l_out.append(action)  # screen negative-reward actions
                print(f"Episode: {episode}, Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")
                next_obs, next_rewards, next_dones, next_infos = \
                    next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info]
                if info['score'] > max_score:  # new high score experienced
                    max_score = info['score']
                    self.agent.memory.clear_alpha()
            rewards, dones, infos = next_rewards, next_dones, next_infos

            # generate valid actions
            next_states = build_state(self.lm, next_obs, infos, prev_obs=obs, prev_acts=action_strs)
            if self.args.lm_top_k:
                for env, info, state, done in zip(eval_envs, infos, next_states, dones):
                    if not done:
                        key = hash(tuple(state[0] + state[1] + state[2]))
                        actions = self.lm.generate(state.state, k=self.args.lm_top_k)
                        l_in, l_out = d_in[key], d_out[key]
                        actions += [action for action in l_in if action not in actions]  # add extra valid
                        actions = [action for action in actions if action and action not in l_out]  # remove invalid
                        valid = [action for action in actions if action in info['valid']]
                        if not valid: valid = ['look', 'inventory', 'task']
                        info['valid'] = valid
            next_valids = [[self.lm.act2ids(a) for a in info['valid']] for info in infos]
            for state, act, rew, next_state, valids, done, transition in zip(states, action_ids, rewards, next_states,
                                                                            next_valids, dones, transitions):
                if act:  # not [] (i.e. reset)
                    transition.append(Transition(state, act, rew, next_state, valids, done))
                    self.agent.observe(transition[-1])  # , is_prior=(rew != 0))
            obs, states, valid_ids = next_obs, next_states, next_valids
            step += 1
          


    def train(self):
        start = time.time()
        obs, rewards, dones, infos, transitions = [], [], [], [], []
        env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)

        train_envs = [CALMScienceWorldEnv(self.args.jar_path, self.args.seed, self.args.task_num, self.args.env_step_limit, self.args.stuck_step_limit,
                                     get_valid=self.args.lm_top_k == 0, simplification_str=self.args.simplification_str, thread_offset=0)
                                     for _ in range(self.args.num_envs)]


        if self.args.lm_dict:
            d_in = defaultdict(list, json.load(open('%s/d_in.json' % self.args.lm_dict, 'r')))
            d_out = defaultdict(list, json.load(open('%s/d_out.json' % self.args.lm_dict, 'r')))
            self.lm.generate_dict = json.load(open('%s/lm.json' % self.args.lm_dict, 'r'))
        for n, env in enumerate(train_envs):
            env.create(n, 0)
            variation = random.choice(self.train_var_nos)
            env.load(variation)
            ob, info = env.reset()
            obs, rewards, dones, infos, transitions = \
                obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]]
        states = build_state(self.lm, obs, infos)
        valid_ids = [[self.lm.act2ids(a) for a in info['valid']] for info in infos]

        episode = 0
        max_score_per_step = []
        for step in range(1, self.args.max_steps + 1):
            # act
            action_ids, action_idxs, action_values = self.agent.act(states, valid_ids, lm=self.lm,
                                                            eps=self.args.eps, alpha=self.args.alpha, k=self.args.eps_top_k)
            action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

            # step with rejection
            next_obs, next_rewards, next_dones, next_infos = [], [], [], []
            for i, (env, action) in enumerate(zip(train_envs, action_strs)):
                if dones[i]:
                    if env.max_score >= max_score:  # put in alpha queue
                        for transition in transitions[i]:
                            self.agent.observe(transition, is_prior=True)

                    self.train_scores.append(infos[i]['score'])
                    
                    print(f"Episode {episode}, Variation {env.variation}")
                    print(f"Totol Steps {step}")
                    print(f"Episode Steps {infos[i]['moves']}")
                    print(f"FPS {round((step * self.args.num_envs) / (time.time() - start), 2)}")
                    print(f"EpisodeScore {infos[i]['score']}")
                    print(f"Max score seen {max_score}" )
                    print()
                    episode += 1
                    if episode < 9:
                        last_10_score = mean(self.train_scores)
                    else:
                        last_10_score = mean(self.train_scores[int(len(self.train_scores)*0.9):])
                    run_history = infos[i]['history']
                    self.bufferedHistorySaverTrain.storeRunHistory(run_history, episode, notes={'last_10_score': last_10_score, 'step': step})
                    self.bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=self.args.max_histories_per_file)                
                    
                    variation = random.choice(self.train_var_nos)
                    env.load(variation)
                    ob, info = env.reset()
                    action_strs[i], obs[i], action_ids[i], transitions[i] = '', '', [], []
                    next_obs, next_rewards, next_dones, next_infos = next_obs + [ob], next_rewards + [0], next_dones + [
                        False], next_infos + [info]
                    continue
                prev_inv, prev_look = infos[i]['inv'], infos[i]['look']
                ob, reward, done, info = env.step(action)
                if self.args.lm_top_k:  # deal with rejection
                    key = hash(tuple(states[i][0] + states[i][1] + states[i][2]))
                    l_in, l_out = d_in[key], d_out[key]
                    actions = infos[i]['valid']
                    rej = 'Input: No known action matches that input.' in ob

                    # while action is invalid, pull another action from CALM generated candidates
                    while not done and rej and len(actions) > 1:
                        if action not in l_out: l_out.append(action)
                        actions.remove(action)
                        action = choice(actions)
                        ob, reward, done, info = env.step(action)
                        rej = 'Input: No known action matches that input.' in ob
                    action_strs[i] = action

                    if not rej and action not in l_in: l_in.append(action)
                    if reward < 0 and action not in l_out: l_out.append(action)  # screen negative-reward actions
                env_steps = infos[i]['moves']
                print(f"Episode: {episode}, Environment: {i}, Step: {step}, Env_Step: {env_steps}, Action: {action}, Reward: {reward}, Done: {done}")
                next_obs, next_rewards, next_dones, next_infos = \
                    next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info]
                if info['score'] > max_score:  # new high score experienced
                    max_score = info['score']
                    self.agent.memory.clear_alpha()
                max_score_per_step.append(max_score)
            rewards, dones, infos = next_rewards, next_dones, next_infos

            # generate valid actions
            next_states = build_state(self.lm, next_obs, infos, prev_obs=obs, prev_acts=action_strs)
            if self.args.lm_top_k:
                for env, info, state, done in zip(train_envs, infos, next_states, dones):
                    if not done:
                        key = hash(tuple(state[0] + state[1] + state[2]))
                        actions = self.lm.generate(state.state, k=self.args.lm_top_k)
                        l_in, l_out = d_in[key], d_out[key]
                        actions += [action for action in l_in if action not in actions]  # add extra valid
                        actions = [action for action in actions if action and action not in l_out]  # remove invalid
                        valid = [action for action in actions if action in info['valid']]
                        if not valid: valid = ['look', 'inventory', 'task']
                        info['valid'] = valid
            next_valids = [[self.lm.act2ids(a) for a in info['valid']] for info in infos]
            for state, act, rew, next_state, valids, done, transition in zip(states, action_ids, rewards, next_states,
                                                                            next_valids, dones, transitions):
                if act:  # not [] (i.e. reset)
                    transition.append(Transition(state, act, rew, next_state, valids, done))
                    self.agent.observe(transition[-1])  # , is_prior=(rew != 0))
            obs, states, valid_ids = next_obs, next_states, next_valids

            loss = self.agent.update()
            print(f'Loss: {loss}')

            if step % self.args.checkpoint_freq == 0:
                json.dump(d_in, open('%s/d_in.json' % self.args.output_dir, 'w'), indent=4)
                json.dump(d_out, open('%s/d_out.json' % self.args.output_dir, 'w'), indent=4)
                json.dump(self.lm.generate_dict, open('%s/lm.json' % self.args.output_dir, 'w'), indent=4)
            
            if step % self.args.eval_freq == 0:
                with torch.no_grad():
                    self.evaluate(step)
        
        self.bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=self.args.max_histories_per_file, forceSave=True)
        self.bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=self.args.max_histories_per_file, forceSave=True)

        with open(os.path.join(self.args.output_dir, "train_episode_score.json"), 'w') as f:
            json.dump(self.train_scores,f)
        
        with open(os.path.join(self.args.output_dir, "eval_episode_score.json"), 'w') as f:
            json.dump(self.eval_scores,f)

        for env in train_envs:
            env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./logs/model')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--stuck_step_limit', default=200, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=1000, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    # logger
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='textgame', type=str)

    # language model
    parser.add_argument('--lm_top_k', default=30, type=int,
                        help='when >0, use lm top-k actions in place of ScienceWorld action detection')
    parser.add_argument('--lm_path', default='gpt2')
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--lm_dict', default='')

    # exploration
    parser.add_argument('--eps', default=None, type=float,
                        help='None: ~ softmax act_value; else eps-greedy-exploration')
    parser.add_argument('--eps_top_k', default=-1, type=int,
                        help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)')
    parser.add_argument('--alpha', default=0, type=float,
                        help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now')
    
    parser.add_argument("--jar_path", type=str)

    parser.add_argument("--task_num", type=int, default=0)
    parser.add_argument("--simplification_str", type=str, default="easy")
    parser.add_argument("--max_eval_episodes", type=int, default=10)
    parser.add_argument("--max_histories_per_file", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = CALMModel(args)
    trainer.train()

if __name__ == "__main__":
    main()
