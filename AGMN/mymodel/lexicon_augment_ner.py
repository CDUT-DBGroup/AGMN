# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from mymodel.transformer import self_TransformerEncoder
from transformers import BertConfig
from .crf import CRF
from transformers import BertModel
from .model import InterModalityUpdate,CNNimgembedding,TransformerRadicalEmbedding

from Modules.TransformerEncoder import TransformerEncoder

class AGMN(nn.Module):
    def __init__(self, data):
        super(AGMN, self).__init__()
        self.dataset = data.dataset
        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.img_weight=data.img_weight
        self.repre_num=data.repre_num
        self.word_emb_dim = data.word_emb_dim
        self.use_char = data.HP_use_char
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.use_count = data.HP_use_count
        self.num_layer = data.HP_num_layer
        self.model_type = data.model_type
        self.use_bert = data.use_bert
        self.dropout = data.dropout
        self.layer_preprocess_sequence = ""
        self.layer_postprocess_sequence = "an"
        # self.drop = nn.Dropout(0.5)
        num_layers = 2
        d_model = 768
        n_head = 8
        self.char_pad_index=0
        feedforward_dim = 2 * 768
        self.hidden_size=128
        self.num_heads=8
        dropout = 0.15
        radical_dropout = 0.2
        self.ff_size = 128*3
        self.max_seq_len=250
        self.ff_activate="relu"
        self.rel_pos_init = 1

        after_norm = True
        pos_embed = None
        dropout_attn = None
        attn_type = 'adatrans'

        self.transformer = self_TransformerEncoder(num_layers, self.hidden_size, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=attn_type == 'transformer', dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.dropout_cnn = nn.Dropout(0.2)
        self.components = TransformerRadicalEmbedding(alphabet=data.word_alphabet, embed_size=200, char_emb_size=128,
                                              filter_nums=50, char_dropout=dropout,
                                                    dropout=radical_dropout,
                                              kernel_sizes= 3, pool_method='max'
                                              , include_word_start_end=False, min_char_freq=1)
        self.add_img_repre=CNNimgembedding(self.hidden_size)
        self.char_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                               dataset=self.dataset,
                                               layer_preprocess_sequence=self.layer_preprocess_sequence,
                                               layer_postprocess_sequence=self.layer_postprocess_sequence,
                                               dropout=self.dropout,

                                               ff_size=self.ff_size,
                                               max_seq_len=self.max_seq_len,
                                               #pe=pe,
                                               ff_activate=self.ff_activate,)
        self.in_fc = nn.Linear(200 * 4, self.hidden_size)


        self.dropout = nn.Dropout(0.5)
        init.xavier_uniform_(self.in_fc.weight.data)
        init.constant_(self.in_fc.bias.data, 0)
        self.layn = nn.LayerNorm(768, elementwise_affine=False)
        self.components_proj = nn.Linear(128, self.hidden_size)
        self.char_proj = nn.Linear(768, self.hidden_size)

        self.img_proj=nn.Linear(768,self.hidden_size)
        self.output = nn.Linear(self.hidden_size , data.label_alphabet_size + 2)

        self.dy = InterModalityUpdate(128,128, 128,  128, 8, 0.5)

        self.cat2gatess_1 = nn.Linear(768, 768)

        self.cat2gatess_2 = nn.Linear(768*2, 768)

        self.cat2gatess_3 = nn.Linear(768, 768)

        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])

        if self.use_char:
            scale = np.sqrt(3.0 / self.word_emb_dim)
            data.pretrain_word_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])

        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)


        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))

        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.word_emb_dim)))


        char_feature_dim = self.word_emb_dim + 4 * self.gaz_emb_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768

        print('total char_feature_dim {}'.format(char_feature_dim))

        self.drop = nn.Dropout(p=data.HP_dropout)

        self.hidden2tag = nn.Linear(768, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.use_bert:
            config = BertConfig.from_json_file(
                'E:/data/bert-base-chinese/bert_config.json')
            self.bert_encoder = BertModel.from_pretrained('E:/data/bert-base-chinese', config=config,ignore_mismatched_sizes=True)
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        if self.gpu:
            self.transformer=self.transformer.cuda()
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()

            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()
            self.in_fc = self.in_fc.cuda()
            self.cat2gatess_3 = self.cat2gatess_3.cuda()
            self.cat2gatess_1 = self.cat2gatess_1.cuda()
            self.cat2gatess_2 = self.cat2gatess_2.cuda()
            self.dy= self.dy.cuda()
            self.drop = self.drop.cuda()
            self.dropout=self.dropout.cuda()
            self.dropout_cnn=self.dropout_cnn.cuda()
            self.components=self.components.cuda()
            self.add_img_repre=self.add_img_repre.cuda()
            self.char_encoder=self.char_encoder.cuda()
            self.components_proj=self.components_proj.cuda()
            self.img_proj=self.img_proj.cuda()
            self.char_proj=self.char_proj.cuda()
            self.output=self.output.cuda()

            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()

    def get_tags(self, gaz_list, word_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input,
                 gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        mask = mask.bool()

        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []
        mask1 = word_inputs.ne(0)
        word_embs = self.word_embedding(word_inputs)

        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1, 1, 1, 1, 1, self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  # (b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  # (b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  # (b,l,4,gl,ce)

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        else:  # use gaz embedding
            gaz_embeds = self.gaz_embedding(layer_gaz)

            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
            else:
                gaz_embeds_d = gaz_embeds

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1, 1, 1, 1, self.gaz_emb_dim)
            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0)  # (b,l,4,g,ge)  ge:gaz_embed_dim

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  # (b,l,4,gn)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  # (b,l,1,1)

            weights = gaz_count.div(count_sum)  # (b,l,4,g)
            weights = weights * 4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights * gaz_embeds  # (b,l,4,g,e)
            gaz_embeds = torch.sum(gaz_embeds, dim=3)  # (b,l,4,e)

        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  # (b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  # (b,l,4,ge)/(b,l,4,1)

        gaz_embeds_cat = gaz_embeds.view(batch_size, seq_len, -1)  # (b,l,4*ge)
        # word_input_cat = torch.cat([word_inputs_d, gaz_embeds_cat], dim=-1)  # (b,l,we+4*ge)

        ### cat bert feature
        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long()
            seg_id=seg_id.cuda()
            #seg_id = torch.zeros(bert_mask.size()).long()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            outputs = outputs[0][:, 1:-1, :]
            outputs=self.dropout(outputs)
            outputs = self.char_proj(outputs)

        gaz_embeds_cat=self.dropout(gaz_embeds_cat)
        gaz_embeds_cat = self.in_fc(gaz_embeds_cat)   #[2, 31, 800]


        img_repre = self.add_img_repre(word_inputs,self.img_weight)
        components_embed = self.components(word_inputs,mask)   #[1, 66, 50]


        char_encoded = self.char_encoder(outputs, gaz_embeds_cat, gaz_embeds_cat, seq_len)
        outputs, _ = self.dy(char_encoded, img_repre, mask, mask)

        """Self-lattice Attention"""

        tags =self.output(outputs)

        return tags, gaz_match

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, word_seq_lengths, layer_gaz, gaz_count,
                                gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bert, bert_mask):

        """表示了每个单词对应的每个标签的得分（score）。"""
        tags, _ = self.get_tags(gaz_list, word_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                                gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)


        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, gaz_list, word_inputs, word_seq_lengths, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                gazchar_mask, mask, batch_bert, bert_mask):

        tags, gaz_match = self.get_tags(gaz_list, word_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                                        gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq, gaz_match






