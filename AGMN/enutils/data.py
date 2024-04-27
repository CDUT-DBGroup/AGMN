# -*- coding: utf-8 -*-

import sys
import numpy as np
from enutils.alphabet import Alphabet
from enutils.functions import *
from enutils.gazetteer import Gazetteer
import collections

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 112
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.gaz_count = {}
        self.gaz_split = {}

        self.biword_count = {}

        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True
        self.HP_use_count = False

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM" 

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        self.use_bigram = True
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.gaz_dropout = 0.5
        self.img_h = 50
        self.img_w = 50
        self.img_weight=None
        self.repre_num=64
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 128
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.dropout = collections.defaultdict(int)
        self.HP_num_layer = 4

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Use          bigram: %s"%(self.use_bigram))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Gaz   alphabet size: %s"%(self.gaz_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Gaz embedding size: %s"%(self.gaz_emb_dim))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Norm     biword emb: %s"%(self.norm_biword_emb))
        print("     Norm     gaz    emb: %s"%(self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s"%(self.gaz_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara     use_gaz: %s"%(self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s"%(self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s"%(self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        #
        with open(input_file, 'r', encoding="utf-8") as in_lines:
            for line in in_lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    label = pairs[-1]
                    self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))


    def build_alphabet(self, input_file):   #char_alphabet和word_alphabet中的实例和索引是一样的，但是它们的用途不同。word_alphabet用于表示单字级别的特征，而char_alphabet用于表示字符级别的特征。在一些模型中，可能会同时使用两种级别的特征，以提高模型的性能。
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        seqlen = 0
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)


                for char in word:
                    self.char_alphabet.add(char)

                seqlen += 1
            else:
                seqlen = 0

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
    """根据一个词典文件，构建一个词典树（gazetteer tree），用来存储和查询一些词语"""
    def build_gaz_file(self, gaz_file):
        ## 调用self.gaz.insert(fin, "one_source")方法，把它插入到词典树中，并标记它的来源为"one_source"。这个方法会把词语按照字符拆分，并按照层级结构存储在树节点中。例如，“北京市"会被拆分为"北”、“京”、"市"三个字符，并依次存储在根节点的子节点、子节点的子节点、子节点的子节点中
        if gaz_file:
            with open(gaz_file, 'r', encoding="utf-8") as fins:
            # fins = open(gaz_file, 'r',encoding="utf-8").readlines()
                for fin in fins:
                    fin = fin.strip().split()[0]
                    if fin:
                        self.gaz.insert(fin, "one_source")
                print ("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print ("Gaz file is None, load nothing")

    """构建一个词典字母表（gazetteer alphabet），用来存储和管理词典树中的词语和它们对应的索引"""
    def build_gaz_alphabet(self, input_file, count=False):
        # in_lines = open(input_file,'r',encoding="utf-8").readlines()
        with open(input_file, 'r', encoding="utf-8") as  in_lines:
            word_list = []
            for line in in_lines:
                if len(line) > 3:
                    word = line.split()[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    word_list.append(word)
                else:
                    w_length = len(word_list)
                    entitys = []
                    for idx in range(w_length):
                        matched_entity = self.gaz.enumerateMatchList(word_list[idx:])  #把从当前位置开始的所有匹配的词典树中的词语返回，并添加到实体列表中
                        entitys += matched_entity
                        for entity in matched_entity:
                            # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                            self.gaz_alphabet.add(entity)
                            index = self.gaz_alphabet.get_index(entity)


                            self.gaz_count[index] = self.gaz_count.get(index,0)  ## initialize gaz count


                    if count:
                        entitys.sort(key=lambda x:-len(x))      #把实体列表按照长度降序排序
                        while entitys:
                            longest = entitys[0]
                            longest_index = self.gaz_alphabet.get_index(longest)
                            self.gaz_count[longest_index] = self.gaz_count.get(longest_index, 0) + 1

                            gazlen = len(longest)
                            for i in range(gazlen):   #对于最长实体中的每个子串（从左到右，从短到长），如果它也在实体列表中，那么从实体列表中移除它。例如，移除"北京"、“天安门”、"北京天安门"等。
                                for j in range(i+1,gazlen+1):
                                    covering_gaz = longest[i:j]
                                    if covering_gaz in entitys:
                                        entitys.remove(covering_gaz)
                                        # print('remove:',covering_gaz)
                    word_list = []
            print("gaz alphabet size:", self.gaz_alphabet.size())

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        print ("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        print("word_emb_dim",self.word_emb_dim)


    def build_gaz_pretrain_emb(self, emb_path):
        print ("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,  self.gaz_emb_dim, self.norm_gaz_emb)
        print("gaz_emb_dim",self.gaz_emb_dim)

    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet,  self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz,self.word_alphabet,   self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet, self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(self.HP_num_layer, input_file, self.gaz, self.word_alphabet,self.char_alphabet, self.gaz_alphabet, self.gaz_count, self.gaz_split,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))
    def generate_img_weight(self,input_file):
        print("build word input img weight...")
        self.img_weight = get_img_weight(input_file,self.word_alphabet,self.img_h, self.img_w)





