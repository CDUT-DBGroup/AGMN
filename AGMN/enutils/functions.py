# -*- coding: utf-8 -*-

import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import BertConfig
import os
import gif2numpy # 或者 import imageio
import torch
from torchvision import transforms
NULLKEY = "-null-"


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_with_gaz(num_layer, input_file, gaz, word_alphabet,char_alphabet,
                           gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length,
                           char_padding_size=-1, char_padding_symbol='</pad>'):
    config = BertConfig.from_json_file(
        'E:/data/bert-base-chinese/bert_config.json')
    #tokenizer 是一个对象，表示一个分词器，它可以把文本分割成词或子词，并把它们映射到一个词表中的索引。
    tokenizer = BertTokenizer.from_pretrained('E:/data/bert-base-chinese', config=config, do_lower_case=True)

    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:   #说明这是一个单词和标签的对
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:  #表示这是一句话的结束标志，对if条件处理好的这一句话进行操作
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):
                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []
                w_length = len(words)
                """三个二维列表，用来存储每个单词的每个位置（开始、中间、结束、单字）的匹配词语的编号、数量和字符编号,其中每个元素是一个长度为4的列表 
                eg: gaz[][] 其中第一个[]表示一句话中char的序号idx，在下图中也就是7，第二个[]表示BMES，也就是所匹配到的单词是属于开始还是结尾等等。
                """
                gazs = [[[] for i in range(4)] for _ in
                        range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                gazs_count = [[[] for i in range(4)] for _ in range(w_length)]

                gaz_char_Id = [[[] for i in range(4)] for _ in
                               range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

                max_gazlist = 0
                max_gazcharlen = 0
                for idx in range(w_length):

                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]

                    if matched_length:
                        max_gazcharlen = max(max(matched_length), max_gazcharlen)

                    for w in range(len(matched_Id)):
                        gaz_chars = []
                        g = matched_list[w]
                        for c in g:
                            gaz_chars.append(word_alphabet.get_index(c))

                        if matched_length[w] == 1:  ## Single
                            gazs[idx][3].append(matched_Id[w])
                            gazs_count[idx][3].append(1)
                            gaz_char_Id[idx][3].append(gaz_chars)
                        else:
                            gazs[idx][0].append(matched_Id[w])  ## Begin   #gaz表示单词
                            gazs_count[idx][0].append(gaz_count[matched_Id[w]])  #gazs_count表示频率，出现次数
                            gaz_char_Id[idx][0].append(gaz_chars)
                            wlen = matched_length[w]
                            gazs[idx + wlen - 1][2].append(matched_Id[w])  ## End
                            gazs_count[idx + wlen - 1][2].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx + wlen - 1][2].append(gaz_chars)
                            for l in range(wlen - 2):
                                gazs[idx + l + 1][1].append(matched_Id[w])  ## Middle
                                gazs_count[idx + l + 1][1].append(gaz_count[matched_Id[w]])
                                gaz_char_Id[idx + l + 1][1].append(gaz_chars)

                    for label in range(4):
                        if not gazs[idx][label]:
                            gazs[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])

                        max_gazlist = max(len(gazs[idx][label]), max_gazlist)

                    matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]  # 词号
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])

                ## batch_size = 1
                for idx in range(w_length):
                    gazmask = []
                    gazcharmask = []
                    wordsmask=[]

                    for label in range(4):      #定义一个变量label_len，用来存储当前单词的当前位置的匹配词语的数量。例如，当idx为3时，label为0时，label_len为1，因为gazs[3][0]中有一个元素；label为1时，label_len为0，因为gazs[3][1]中没有元素。
                        label_len = len(gazs[idx][label])
                        count_set = set(gazs_count[idx][label])
                        if len(count_set) == 1 and 0 in count_set:
                            gazs_count[idx][label] = [1] * label_len

                        mask = label_len * [0]
                        mask += (max_gazlist - label_len) * [1]

                        gazs[idx][label] += (max_gazlist - label_len) * [0]  ## padding
                        gazs_count[idx][label] += (max_gazlist - label_len) * [0]  ## padding

                        char_mask = []
                        for g in range(len(gaz_char_Id[idx][label])):
                            glen = len(gaz_char_Id[idx][label][g])
                            charmask = glen * [0]
                            charmask += (max_gazcharlen - glen) * [1]
                            char_mask.append(charmask)
                            gaz_char_Id[idx][label][g] += (max_gazcharlen - glen) * [0]
                        gaz_char_Id[idx][label] += (max_gazlist - label_len) * [[0 for i in range(max_gazcharlen)]]
                        char_mask += (max_gazlist - label_len) * [[1 for i in range(max_gazcharlen)]]

                        gazmask.append(mask)
                        gazcharmask.append(char_mask)
                    layergazmasks.append(gazmask)  #用于存储对gazs和gaz_char_Id进行masking后的结果
                    gazchar_masks.append(gazcharmask)

                texts = ['[CLS]'] + words + ['[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)  #，texts和bert_text_ids就可以作为BERT模型的输入，进行后续的处理。

                instence_texts.append([words, chars, gazs, labels])
                instence_Ids.append(
                    [word_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,
                     gazchar_masks, bert_text_ids])

            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids

def convert_to_tensor(array,img_h, img_w):
  # 将数组转换为张量
  tensor = torch.from_numpy(array)
  # 将张量转换为三维的，如果是灰度图，第一维度为1
  if len(tensor.shape) == 2:
    tensor = tensor.unsqueeze(0)
  # 将张量转换为50*50的大小
  resize = transforms.Resize((img_h, img_w))
  tensor = resize(tensor)
  tensor = torch.unsqueeze(tensor, dim=-1)
  return tensor

def get_img_weight(input_file,word_alphabet,img_h, img_w):

    # 定义一个空字典，用来存储文件名和张量
    img_weight = {}
    # 遍历文件夹中的所有文件
    for file in os.listdir(input_file):
        # 如果文件是gif格式的
        if file.endswith(".gif"):
            # 获取文件的完整路径
            file_path = os.path.join(input_file, file)
            # 读取gif文件，并将其转换为numpy数组
            np_frames, _, _ = gif2numpy.convert(file_path)  # 或者 np_frames = imageio.mimread(file_path)
            # 取第一帧作为代表，或者你可以选择其他帧或者平均所有帧
            np_frame = np_frames[0]
            # 将numpy数组转换为50*50的张量
            tensor = convert_to_tensor(np_frame,img_h, img_w)
            # 去掉文件名的后缀名
            file_name = os.path.splitext(file)[0]
            # 将文件名和张量存储在字典中
            img_weight[file_name] = tensor
    for key,_ in img_weight.items():
        index=word_alphabet.get_index(key)
        img_weight[key]=index
    return img_weight


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=50, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        print(embedding_path)
        if embedding_path == "E:/data/gigaword_chn.all.a2b.uni.ite50.vec":
            embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
        else:
            embedd_dict, embedd_dim = load_pretrain_emb1(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    print(embedd_dim)
    print(word_alphabet.size())
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb[0, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        #next(file) #跳过第一行
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            # t=len(tokens)
            # if len(tokens) != 201 :
            #     continue
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
                print(embedd_dim)
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim
def load_pretrain_emb1(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            # t=len(tokens)
            if len(tokens) != 201 :   #如果是腾讯的则改为201，如果是bi则改为51
                continue
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

