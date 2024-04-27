import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from enutils.alphabet import Alphabet
from mymodel.transformer import self_TransformerEncoder
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.embeddings import StaticEmbedding
import torch.nn.init as init
radical_path = 'E:/data/chaizi-master/char_info.txt'
def get_char_info():
    char_info = dict()
    with open(radical_path, 'r',encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            char, info = line.split('\t', 1)
            char_info[char] = info.replace('\n', '').split('\t')
    return char_info


"""将汉字转化为部首
"""
def char2radical(c):
    char_info= get_char_info()
    if c in char_info.keys():
        c_info = char_info[c]

        return list(c_info)
    return ['○']




def is_chinese(char):
    if len(char) != 1:
        return False
    elif ord("\u4e00") <= ord(char) <= ord("\u9fff"):
        return True
    else:
        return False
def _construct_radical_vocab_from_vocab(char_alphabet: Alphabet, min_freq: int = 1, include_word_start_end=True):
    r"""


    :param vocab: 从vocab
    :param min_freq:
    :param include_word_start_end
    :return:
    """
    word_list = char_alphabet.get_content()

    radical_vocab = Alphabet('radical')
    word_iter = iter(word_list["instance2index"].items())
    next(word_iter, None)
    for char, index in word_iter:
        #print("输入函数的char：",char)
        if is_chinese(char):
          radical_vocab.add(char2radical(char))
    if include_word_start_end:
        radical_vocab.add(['<bow>', '<eow>'])
    return radical_vocab


class TransformerRadicalEmbedding(nn.Module):
    def __init__(self, alphabet: Alphabet, embed_size: int = 200, char_emb_size: int = 50, char_dropout: float = 0,
                 dropout: float = 0, filter_nums: int = 30, kernel_sizes: int = 3,
                 pool_method: str = 'max', activation='relu', min_char_freq: int = 2, pre_train_char_embed: str = None,
                 requires_grad: bool = True, include_word_start_end: bool = True):

        super(TransformerRadicalEmbedding, self).__init__()
        self.transformer = self_TransformerEncoder(2, 128, 8, 2*128, dropout,
                                              after_norm= True, attn_type='transformer',
                                              scale= 'transformer', dropout_attn=None,
                                              pos_embed=None)
        if kernel_sizes:
            assert kernel_sizes % 2 == 1, "Only odd kernel is allowed."

        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method

        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")
        word_list = alphabet.get_content()

        self.radical_alphabet = _construct_radical_vocab_from_vocab(alphabet, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)

        self.char_pad_index = 0
        # Z最长部首数量
        max_radical_nums = max(map(lambda x: len(char2radical(x)),word_list["instances"]))
        print("max_radical_nums",max_radical_nums)
        if include_word_start_end:
            max_radical_nums += 2

        self.register_buffer('chars_to_radicals_embedding', torch.full((len(word_list["instances"])+2, max_radical_nums),  #chars_to_radicals_embedding存储从字符到部首的映射的张量
                                                                    fill_value=self.char_pad_index, dtype=torch.long))

        self.register_buffer('word_lengths', torch.zeros(len(word_list["instances"])+2).long())
        for word, index in word_list["instance2index"].items():

            word = char2radical(word)
            if include_word_start_end:
                word = ['<bow>'] + word + ['<eow>']


            self.chars_to_radicals_embedding[index, :len(word)] = \
                torch.LongTensor([self.radical_alphabet.get_index(c) for c in word])
            self.word_lengths[index] = len(word)
        #self.char_embedding = get_embeddings((len(self.radical_alphabet), char_emb_size))
        #self.char_embedding = StaticEmbedding(self.radical_vocab,model_dir_or_name='/home/ws/data/gigaword_chn.all.a2b.uni.ite50.vec')
        self.char_embedding = nn.Embedding(len(self.radical_alphabet)+2, char_emb_size)
        self.drop = nn.Dropout(0.2)


    def forward(self, words,mask):
        r"""

        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, embed_size]
        """
        chars = self.chars_to_radicals_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]# torch.Size([1, 26, 2])
        chars = self.char_embedding(chars)  # 得到batch_size x max_len x max_word_len x embed_size
        chars = self.drop(chars)
        chars = torch.mean(chars, dim=-2, keepdim=False)
        conv_chars=self.transformer(chars,mask)
        return conv_chars




# def ImageEmbeding(char_input,img_weight):  #batch_size=1
#     result = []
#     for b in char_input:
#         temp = []
#         temp.append(img_weight.get(b, torch.zeros(
#             [50, 50, 1])))  # Get the value from A or zero_tensor if the key does not exist
#         result.append(temp)  # Append the temporary list to the result list
#     result = [torch.cat(x, dim=0) for x in result]  # Concatenate each sublist to a tensor
#     result = torch.stack(result)  # Use torch.stack instead of torch.tensor
#     # print("result.size()", result.size())
#
#     return result

def ImageEmbeding(char_input, img_weight):  #batch_size>1
    result = []
    # Split the char_input into batch_size sub-tensors
    char_input = torch.split(char_input, 1, dim=0)
    for b in char_input:
        temp = []
        # Squeeze the sub-tensor to remove the first dimension
        b = torch.squeeze(b,0)
        for c in b:
            # Get the image weight for each character
            temp.append(img_weight.get(c.item(), torch.zeros([50, 50, 1])))
        # Stack the sub-list to a tensor
        temp = torch.stack(temp)
        # Append the tensor to the result list
        result.append(temp)
    # Stack the result list to a tensor
    result = torch.stack(result)
    return result
class CNNimgembedding(nn.Module):
    def __init__(self,hidden_size):
        super(CNNimgembedding, self).__init__()
        self.activation = F.relu
        self.dropout_cnn = nn.Dropout(0.2)
        self.fc = nn.Linear(144, hidden_size)
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3d2 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Batchnorm= nn.BatchNorm1d(144,track_running_stats=False)
        self.conv2d_list = nn.ModuleList()
        for i in range(150):
            self.conv2d_list.append(nn.Conv2d(in_channels=i+1, out_channels=i+1, kernel_size=3, stride=1, padding=1))
    def forward(self,char_input,img_weight):

        embed = ImageEmbeding(char_input,img_weight)

        #embed = torch.unsqueeze(embed, dim=0)
        embed = embed.permute(0, 4, 1, 2, 3)  # （batch_size, channels, depth, height, width)
        drop = self.dropout_cnn(embed)
        drop = drop.to('cuda')
        drop = self.conv3d1(drop)  #[1, 4, 5, 50, 50])
        drop = self.conv3d2(drop)  #[1, 1, 5, 50, 50])
        drop = drop.permute(0, 1, 2, 3, 4).contiguous().view(drop.size(0), -1, drop.size(3), drop.size(4))  #[1, 5, 50, 50]
        conv2d_1 = self.conv2d_list[drop.size(1)-1]
        conv = conv2d_1(drop)  #[1, 5, 50, 50]
        conv = self.maxpool2d(conv)  #[1, 5, 25, 25]
        conv2d_2 = self.conv2d_list[drop.size(1) - 1]
        conv = conv2d_2(conv)  # [1, 5, 50, 50]
        conv = self.maxpool2d(conv)  # [1, 5, 12, 12]
        drop = conv.permute(0, 1, 2, 3).contiguous().view(conv.size(0), conv.size(1), -1)
        # drop = drop.permute(0, 2, 1)  #Batchnorm的输入是（batch，channel，len）
        # fea_repre = self.Batchnorm(drop)
        # fea_repre=fea_repre.permute(0, 2, 1)
        fea_repre=self.activation(drop)
        fea_repre = self.fc(fea_repre)
        return fea_repre


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)

class ReshapeBatchNorm(nn.Module):
    def __init__(self, feat_size, affine=True):
        super(ReshapeBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(feat_size, affine=affine)

    def forward(self, x):
        assert(len(x.shape) == 3)
        batch_size, num, _ = x.shape
        x = x.view(batch_size * num, -1)
        x = self.bn(x)
        return x.view(batch_size, num, -1)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        #self.fusion = Fusion()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        out = self.lin1(self.drop(v_mean * q_mean))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out

class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = nn.Linear(v_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)

        self.interBlock = InterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        q = self.q_lin(self.drop(q))
        for i in range(self.num_block):
            v, q = self.interBlock(v, q, v_mask, q_mask)
            v, q = self.intraBlock(v, q, v_mask, q_mask)
        return v,q

class MultiBlock(nn.Module):
    """
    Multi Block Inter-/Intra-modality
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_block = num_block

        blocks = []
        blocks.append(InterModalityUpdate(v_size, q_size, output_size, num_head, drop))
        blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        for i in range(num_block - 1):
            blocks.append(InterModalityUpdate(output_size, output_size, output_size, num_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        for i in range(self.num_block):
            v, q = self.multi_blocks[i*2+0](v, q, v_mask, q_mask)
            v, q = self.multi_blocks[i*2+1](v, q, v_mask, q_mask)
        return v,q
# class InterModalityUpdate(nn.Module):  #类似于transformer  但是只有q和v输入
#     """
#     Inter-modality Attention Flow
#     """
#     def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
#         super(InterModalityUpdate, self).__init__()
#         self.v_size = v_size
#         self.q_size = q_size
#         self.output_size = output_size
#         self.num_head = num_head
#         #output_size 要乘 3 的原因是，这个类是用于实现一个多头注意力机制的，其中每个头需要三个向量：查询（query），键（key）和值（value）。所以，输出的维度要乘以 3，然后再分成三个部分，分别对应查询，键和值。这样做的好处是，可以用一个线性层同时得到三个向量，而不用定义三个不同的线性层
#         self.v_lin = nn.Linear(v_size, output_size * 3)
#         self.q_lin = nn.Linear(q_size, output_size * 3)
#         """output_size + q_size 的原因是，这个类是用于实现一个跨模态的注意力更新的，其中每个模态的特征向量要和另一个模态的注意力加权值向量进行拼接，
#         然后再用一个线性层得到新的特征向量。所以，输入的维度要加上另一个模态的输出维度，然后再映射到输出维度。这样做的好处是，
#         可以保留原始的特征信息，同时融合另一个模态的注意力信息。
#         是跨模态的注意力的原因是，这个类是用于处理多模态的数据的，比如视觉和语言。多模态的数据通常有不同的特征空间和分布，
#         所以需要一种机制来建立不同模态之间的联系和对齐。跨模态的注意力就是一种这样的机制，它可以根据一个模态的查询向量，计算另一个模态的键向量的相关性，
#         然后用另一个模态的值向量对查询向量进行更新。这样做的好处是，可以让不同模态的特征互相影响和补充，提高多模态任务的性能。
#         """
#         self.v_output = nn.Linear(output_size + v_size, output_size)   #用于将视觉特征和视觉更新特征拼接后映射为输出特征
#         self.q_output = nn.Linear(output_size + q_size, output_size)  #用于将问题特征和问题更新特征拼接后映射为输出特征
#
#         self.relu = nn.ReLU()
#         self.drop = nn.Dropout(drop)
#
#     """用一个跨模态的注意力机制来更新v和q的特征向量。它的输入是 v, q, v_mask, q_mask，其中 v 是一个三维的张量，
#     表示一批视觉特征向量，q 是一个三维的张量，表示一批语言特征向量，v_mask 是一个二维的张量，表示v特征中的有效位置，
#     q_mask 是一个二维的张量，表示语言特征中的有效位置。它的输出是 updated_v 和 updated_q，也是两个三维的张量，
#     表示经过注意力更新后的v和q特征向量。
#     """
#     def forward(self, v, q, v_mask, q_mask):
#         """
#         v: visual feature      [batch, num_obj, feat_size]
#         q: question            [batch, max_len, feat_size]
#         v_mask                 [batch, num_obj]
#         q_mask                 [batch, max_len]
#         """
#         batch_size, num_obj = v_mask.shape
#         _         , max_len = q_mask.shape
#         # 将视觉特征（v）和问题特征（q）分别映射为三倍输出维度大小的张量,按照三等分切分为查询向量（q）、键向量（k）和值向量（v）
#         v_trans = self.v_lin(self.drop(self.relu(v)))
#         q_trans = self.q_lin(self.drop(self.relu(q)))
#         # 然后根据视觉掩码（v_mask）和问题掩码（q_mask），将无效对象或单词对应的查询向量、键向量和值向量置为0
#         v_trans = v_trans * v_mask.unsqueeze(2)
#         q_trans = q_trans * q_mask.unsqueeze(2)
#         #
#         v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
#         q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
#         # 然后根据多头自注意力的头数（self.num_head），将查询向量、键向量和值向量分别切分为多个小张量
#         vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
#         vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
#         vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
#         qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
#         qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
#         qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
#
#         for i in range(self.num_head):
#             vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
#             qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
#             # 内积并将填充对象/单词注意力设置为负无穷大并通过隐藏维度的平方根进行标准化
#             q2v = (vq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
#             v2q = (qq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
#             # softmax attention
#             interMAF_q2v = F.softmax(q2v, dim=2) #[batch, num_obj, max_len]
#             interMAF_v2q = F.softmax(v2q, dim=2)
#
#             v_update = interMAF_q2v @ qv_slice if (i==0) else torch.cat((v_update, interMAF_q2v @ qv_slice), dim=2)
#             q_update = interMAF_v2q @ vv_slice if (i==0) else torch.cat((q_update, interMAF_v2q @ vv_slice), dim=2)
#
#         cat_v = torch.cat((v, v_update), dim=2)  #将原始的视觉特征（v）和视觉更新特征（v_update）在第二个维度上进行拼接
#         cat_q = torch.cat((q, q_update), dim=2)
#         updated_v = self.v_output(self.drop(cat_v))
#         updated_q = self.q_output(self.drop(cat_q))
#         return updated_v, updated_q
class InterModalityUpdate(nn.Module):  #类似于transformer  但是只有q和v输入
    """
    Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, k_size,output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.k_size = k_size
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)
        self.k_lin = nn.Linear(k_size, output_size * 3)

        self.v_output = nn.Linear(output_size + v_size, output_size)
        self.q_output = nn.Linear(output_size + q_size, output_size)
        self.k_output = nn.Linear(output_size + k_size + v_size, output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)


    def forward(self, v, q, k ,v_mask, q_mask, k_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        _         , max_len_k = k_mask.shape

        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        k_trans = self.k_lin(self.drop(self.relu(k)))

        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        k_trans = k_trans * k_mask.unsqueeze(2)
        #
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        k_k, k_q, k_v = torch.split(k_trans, k_trans.size(2) // 3, dim=2)


        vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
        vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
        qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)

        kk_set = torch.split(k_k, k_k.size(2) // self.num_head, dim=2)
        kq_set = torch.split(k_q, k_q.size(2) // self.num_head, dim=2)
        kv_set = torch.split(k_q, k_v.size(2) // self.num_head, dim=2)

        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            kk_slice, kq_slice, kv_slice = kk_set[i], kq_set[i], kv_set[i]  # [batch, max_len, feat_size]
            # 内积并将填充对象/单词注意力设置为负无穷大并通过隐藏维度的平方根进行标准化,计算**查询（Query）和键（Key）**之间的点积
            q2v = (vq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            v2q = (qq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            k2v = (kq_slice @ kk_slice.transpose(1,2)).masked_fill(k_mask.unsqueeze(1).expand([batch_size, num_obj, max_len_k]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)


            interMAF_q2v = F.softmax(q2v, dim=2) #[batch, num_obj, max_len]
            interMAF_v2q = F.softmax(v2q, dim=2)
            interMAF_k2v = F.softmax(k2v, dim=2)

            vq_update = interMAF_q2v @ qv_slice if (i==0) else torch.cat((vq_update, interMAF_q2v @ qv_slice), dim=2)
            q_update = interMAF_v2q @ vv_slice if (i==0) else torch.cat((q_update, interMAF_v2q @ vv_slice), dim=2)
            vk_update = interMAF_k2v @ kv_slice if (i==0) else torch.cat((vk_update, interMAF_k2v @ kv_slice), dim=2)

        cat_v = torch.cat((v, vq_update), dim=2)
        #cat_q = torch.cat((q, q_update), dim=2)
        cat_v2qk = torch.cat((v, vq_update,vk_update), dim=2)

        updated_v = self.v_output(self.drop(cat_v))
        #updated_q = self.q_output(self.drop(cat_q))
        updated_V = self.k_output(self.drop(cat_v2qk))

        return updated_V, updated_v

class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)


    def forward(self, v, q, v_mask, q_mask):
            """
            v: visual feature      [batch, num_obj, feat_size]
            q: question            [batch, max_len, feat_size]
            v_mask                 [batch, num_obj]
            q_mask                 [batch, max_len]
            """
            batch_size, num_obj = v_mask.shape
            _         , max_len = q_mask.shape
            # conditioned gating vector
            v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
            q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
            v4q_gate = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1) #[batch, 1, feat_size]
            q4v_gate = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))).unsqueeze(1) #[batch, 1, feat_size]

            # key, query, value
            v_trans = self.v_lin(self.drop(self.relu(v)))
            q_trans = self.q_lin(self.drop(self.relu(q)))
            # mask all padding object/word features
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
            # split for different use of purpose
            v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
            q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
            # apply conditioned gate
            new_vq = (1 + q4v_gate) * v_q
            new_vk = (1 + q4v_gate) * v_k
            new_qq = (1 + v4q_gate) * q_q
            new_qk = (1 + v4q_gate) * q_k

            # apply multi-head
            vk_set = torch.split(new_vk, new_vk.size(2) // self.num_head, dim=2)
            vq_set = torch.split(new_vq, new_vq.size(2) // self.num_head, dim=2)
            vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
            qk_set = torch.split(new_qk, new_qk.size(2) // self.num_head, dim=2)
            qq_set = torch.split(new_qq, new_qq.size(2) // self.num_head, dim=2)
            qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
            # multi-head
            for i in range(self.num_head):
                vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
                qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
                # calculate attention
                v2v = (vq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
                q2q = (qq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
                dyIntraMAF_v2v = F.softmax(v2v, dim=2)
                dyIntraMAF_q2q = F.softmax(q2q, dim=2)

                v_update = dyIntraMAF_v2v @ vv_slice if (i==0) else torch.cat((v_update, dyIntraMAF_v2v @ vv_slice), dim=2)
                q_update = dyIntraMAF_q2q @ qv_slice if (i==0) else torch.cat((q_update, dyIntraMAF_q2q @ qv_slice), dim=2)
            # update
            updated_v = self.v_output(self.drop(v + v_update))
            updated_q = self.q_output(self.drop(q + q_update))
            return updated_v, updated_q
