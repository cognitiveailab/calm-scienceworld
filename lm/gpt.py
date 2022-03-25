import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.modeling_utils import *
from transformers.generation_beam_search import BeamHypotheses
from jericho.util import clean
from jericho.defines import ILLEGAL_ACTIONS, NO_EFFECT_ACTIONS

from .base_lm import BaseLM, device


class GPT2LM(BaseLM):
    def load_model(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.generate_dict = {}
        self.model.eval()
        self.model.to(device)

    def load_tokenizer(self,tokenizer_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2' if tokenizer_path is None else tokenizer_path)
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})
        self.model.resize_token_embeddings(len(self.tokenizer)) 

    def act2ids(self, act):
        ret = self.tokenizer.encode(clean(act), add_prefix_space=True)
        if not ret: ret = [0]
        return ret

    def sent2ids(self, sent, maxlen=512):
        ret = self.tokenizer.encode(clean(sent))
        if len(ret) > maxlen:
            ret = ret[-maxlen:]
        if not ret: ret = [0]
        return ret

    def generate(self, input, k, mask_out=ILLEGAL_ACTIONS + NO_EFFECT_ACTIONS, key=None):
        input_ids = self.sent2ids(input) if isinstance(input, str) else input
        if key is None:
            key = hash((tuple(input_ids), k))
        if key in self.generate_dict:
            return self.generate_dict[key]
        input_len = len(input_ids)
        input_ids = torch.tensor([input_ids]).to(device)
        mask_out = [self.tokenizer.encode(' ' + w)[0] for w in mask_out]
        outputs = self.model.generate(
            input_ids,
            do_sample=False,
            num_beams=min(k * 2, 40),
            num_beam_groups=min(k * 2, 40),
            num_return_sequences=k,
            max_length=input_len + 10,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            mask_out=mask_out,
            diversity_penalty = 50.0
        )
        actions = [self.tokenizer.decode(output[input_len:]).split('[SEP]')[0].strip().lower() for output in outputs]
        actions = list(set(actions))
        self.generate_dict[key] = actions
        return actions

    def score(self, input, acts):
        input_ids = self.sent2ids(input) if isinstance(input, str) else input
        input_len = len(input_ids)
        input_ids = torch.tensor([input_ids]).to(device)
        scores = []
        for act in acts.copy():
            if isinstance(act, str):
                act = self.act2ids(act) + self.tokenizer.sep_token
            act_tensor = torch.tensor([act]).to(device)
            example = torch.cat((input_ids, act_tensor), axis=1)
            with torch.no_grad():
                predictions = self.model(example)[0][0][input_len - 1:-1]
            log_p = torch.nn.functional.log_softmax(predictions, dim=-1)
            scores.append(log_p[range(len(act)), act].sum().item())
        return scores



