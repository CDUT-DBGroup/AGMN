import math
from enutils.getembedding import get_embedding
import torch
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from torch import nn

from Modules.MyDropout import MyDropout

"""一个PyTorch的模块，使用XL的位置编码用于实现自注意力机制 (Self-Attention)，加入了randomAttention"""
class AdaptSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 scaled=True, max_seq_len=-1,
                 attn_dropout=None,
                 use_pytorch_dropout=True, dataset='weibo'):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len

        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)

        self.w_r = nn.Linear(self.hidden_size, self.per_head_size)
        # self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        # self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(self.num_heads, self.per_head_size)))
        """r_w_bias这个偏置参数会被加到相对位置编码向量上，用来表示相对于查询向量q的位置信息"""
        self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(self.num_heads, self.per_head_size)))

        if self.use_pytorch_dropout:
            self.dropout = nn.Dropout(attn_dropout)
        else:
            self.dropout = MyDropout(attn_dropout)


        if dataset == 'weibo':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 320, 320), requires_grad=True)
        if dataset == 'msra':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 477, 477), requires_grad=True)
        if dataset == 'resume':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 344, 344), requires_grad=True)
        if dataset == 'ontonotes':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 477, 477), requires_grad=True)
        if dataset == 'ecommerce':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 398, 398), requires_grad=True)

        nn.init.kaiming_normal_(self.randomAttention, a=math.sqrt(5))

    def forward(self, query, key, value, seq_len):  #query: 一个形状为(batch, max_seq_len, hidden_size)的张量

        query = self.w_q(query)
        value = self.w_v(value)

        #rel_pos_embedding = self.w_r(rel_pos_embedding)
        #print(rel_pos_embedding.size())
        batch = key.size(0)
        max_seq_len = key.size(1)

        rel_pos_embedding = get_embedding(max_seq_len, self.hidden_size, rel_pos_init=0).cuda()
        #rel_pos_embedding = get_embedding(max_seq_len, self.hidden_size, rel_pos_init=0).cuda() #生成一个形状为(max_seq_len, max_seq_len, hidden_size)的张量
        rel_pos_embedding = self.w_r(rel_pos_embedding)
        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        #rel_pos_embedding = torch.reshape(rel_pos_embedding,[batch, max_seq_len, max_seq_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)  #将值向量value沿着第1和第2个维度进行转置，使得形状变为(batch, n_head, seq_len, d_head)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        #key = key.transpose(-1, -2)  #将键向量key沿着最后一个和倒数第二个维度进行转置，使得形状变为(batch, n_head, d_head, key_len)
        rw_head_q = query + self.r_r_bias[:, None]  # 偏置参数r_r_bias表示相对于键向量k的位置信息
        """根据指定的公式计算两个张量rw_head_q和k的乘积，并求和,将rw_head_q的第三个维度和k的第三个维度相乘，并在第四个维度上求和，
        得到一个形状为(batch_size, self.n_head, max_len, max_len)的张量AC。这个张量表示每个位置与其他位置之间的注意力得分"""
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, key])  # b x n x l x d,
        """根据指定的公式计算两个张量self.r_w_bias和pos_embed的乘积，并求和。这个公式表示将self.r_w_bias的第一个维度和pos_embed的第一个维度相乘，
        并在第二个维度上求和，得到一个形状为(self.n_head, max_len * 2)的张量D_。这个张量表示每个头的偏置参数与相对位置编码向量的乘积，"""
        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, rel_pos_embedding)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        """根据指定的公式计算两个张量q和pos_embed的乘积，并求和。这个公式表示将q的第四个维度和pos_embed的第二个维度相乘，并在第二个维度上求和，
        得到一个形状为(batch_size, self.n_head, max_len, max_len * 2)的张量B_。这个张量表示每个位置的查询向量与相对位置编码向量的乘积"""
        B_ = torch.einsum('bnqd,ld->bnql', query, rel_pos_embedding)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        """根据指定的公式计算两个张量k和pos_embed的乘积，并求和。这个公式表示将k的第四个维度和pos_embed的第二个维度相乘，并在第二个维度上求和，
        得到一个形状为(batch_size, self.n_head, max_len, max_len * 2)的张量E_。这个张量表示每个位置的键向量与相对位置编码向量的乘积"""
        E_ = torch.einsum('bnqd,ld->bnql', key, rel_pos_embedding)  # bsz x head x max_len x 2max_len, key对relative的bias
        """这个张量表示每个位置与其他位置之间的注意力得分中由查询向量q和偏置参数u决定的部分"""
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE  # 每个位置与其他位置之间的完整的注意力得分

        #B
        # rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2) #使得形状变为(batch, n_head, seq_len, d_head, key_len)
        # """将查询向量query的形状变为(batch, self.num_heads, max_seq_len, 1, self.per_head_size)。这样可以方便地与偏置参数v进行加法运算。"""
        # query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # """将查询向量query_for_b和偏置参数self.v相加，得到一个新的张量query_for_b_and_v_for_d。这个偏置参数self.v是一个形状为(n_head, d_head)的张量，
        # 表示每个头的偏置参数v。为了与查询向量query_for_b的形状匹配，它使用view函数在第0个、第2个和第3个维度上增加了三个空维度，使得形状变为(1, self.num_heads, 1, 1, self.per_head_size)。
        # 这个张量表示每个位置经过偏置参数v调整后的查询向量信息，它的形状也是(batch, self.num_heads, max_seq_len, 1, self.per_head_size)"""
        # query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        # """矩阵乘法，得到一个形状为(batch, self.num_heads, max_seq_len, 1, key_len)的张量B_D。然后使用squeeze函数，将张量B_D中的第3个维度（即大小为1的维度）去掉，
        # 使得形状变为(batch, self.num_heads, max_seq_len, key_len)。这个张量表示每个位置与其他位置之间的注意力得分中由查询向量q和相对位置编码向量决定的部分B。"""
        # B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        #
        # print("B_D shape:", B_D.shape)
        # """将张量A_C和B_D相加，并且加上一个随机注意力矩阵self.randomAttention，得到一个新的张量attn_score_raw。
        # 这个随机注意力矩阵self.randomAttention是一个形状为(batch, n_head, max_seq_len * 2, max_seq_len * 2)的张量，
        # 表示每个位置与其他位置之间的随机注意力得分。为了与A_C和B_D的形状匹配，它使用切片操作只取前max_seq_len个位置的信息，
        # 使得形状变为(batch, n_head, max_seq_len, max_seq_len)。这个张量表示每个位置与其他位置之间的完整的注意力得分
        # A_C: 这是一个张量，表示每个位置与其他位置之间的注意力得分中由查询向量q和键向量k决定的部分。
        # B_D: 这是一个张量，表示每个位置与其他位置之间的注意力得分中由查询向量q和相对位置编码向量决定的部分。
        # self.randomAttention: 这是一个张量，表示每个位置与其他位置之间的随机注意力得分。"""
        # attn_score_raw = A_C + B_D + self.randomAttention[:, :, :max_seq_len, :max_seq_len]




        mask = seq_len_to_mask(torch.tensor([seq_len])).bool().unsqueeze(1).unsqueeze(1).cuda()
        #mask = seq_len_to_mask(torch.tensor([seq_len])).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        return result

    def _shift(self, BD):  # 用来将张量沿着最后一个维度向右移动一位，并在最左边补零，相当于将正负距离转换为绝对距离。
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD

    def _transpose_shift(self, E):  # 用来将张量沿着倒数第二个和最后一个维度进行转置，并调用self._shift函数进行移动，相当于将键向量k的位置信息转换为查询向量q的位置信息。
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

        转换为
          0  -10   -200
          1   00   -100
          2   10    000


        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len) * 2 + 1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1, -2)  # bsz x n_head x max_len x max_len

        return E