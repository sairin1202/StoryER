import torch
import torch.nn.functional as F
from transformers import  LEDForConditionalGeneration

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.longformer = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
        self.score_predictor = torch.nn.Linear(768, 1)
        self.ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=1)
        self.aspect_confidence_predictor = torch.nn.Linear(768, 10)
        self.aspect_rating_predictor = torch.nn.Linear(768, 10)

    def shift_tokens_right(self, input_ids, pad_token_id):
        """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
            This is taken directly from modeling_bart.py
        """
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def forward(self, input_ids, attention_masks, trg_input_ids=None):
        # construct fake input when overall score prediction
        if trg_input_ids is None:
            trg_input_ids = torch.zeros(input_ids.size(0), 1).long().cuda()
        decoder_input_ids = self.shift_tokens_right(trg_input_ids, 1)
        # add global mask
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        self.output = self.longformer(input_ids=input_ids, attention_mask=attention_masks, global_attention_mask=global_attention_mask, decoder_input_ids=decoder_input_ids, output_attentions=True, output_hidden_states=True, return_dict=True)
        lm_logits = self.output[0]
        gen_loss = self.ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), trg_input_ids.view(-1))

        overall_score_pred = self.score_predictor(self.output.encoder_last_hidden_state[:,0,:])
        aspect_conf_pred = self.aspect_confidence_predictor(self.output.encoder_last_hidden_state[:,0,:])
        aspect_score_pred = self.aspect_rating_predictor(self.output.encoder_last_hidden_state[:,0,:])
        return F.sigmoid(overall_score_pred), aspect_conf_pred, F.sigmoid(aspect_score_pred), gen_loss


def get_model():
    model = Model()
    return model



