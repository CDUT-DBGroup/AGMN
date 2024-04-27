# -*- coding: utf-8 -*-

"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.__name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        if not self.label:
            self.add(self.UNKNOWN)
    """用来清空字母表中的实例和索引，并重置属性"""

    def __len__(self):
        # 返回instances列表的长度
        return len(self.instances)
    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
    """用来向字母表中添加一个新的实例。如果实例不存在于字典中，那么会把它添加到列表和字典中，并更新下一个可用的索引。"""
    # def add(self, instance):
    #     if instance not in self.instance2index:
    #         self.instances.append(instance)
    #         self.instance2index[instance] = self.next_index
    #         self.next_index += 1

    def add(self, instance):
        if isinstance(instance, str):  # 判断输入是否是字符串
            if instance not in self.instance2index:
                self.instances.append(instance)
                self.instance2index[instance] = self.next_index
                self.next_index += 1
        elif isinstance(instance, list):  # 判断输入是否是列表
            for item in instance:  # 遍历列表中的每个元素
                if item not in self.instance2index:
                    self.instances.append(item)
                    self.instance2index[item] = self.next_index
                    self.next_index += 1

    """根据实例获取它对应的索引。如果实例存在于字典中，那么返回它的索引；如果不存在，并且字母表可以增长，那么调用add方法添加它，并返回它的索引；如果不存在，并且字母表不可以增长，那么返回未知实例的索引。"""
    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    """根据索引获取它对应的实例。如果索引为0，那么返回None；如果索引在列表范围内，那么返回列表中对应位置的实例；如果索引超出列表范围，那么打印警告信息，并返回列表中第一个实例。"""
    def get_instance(self, index):
        if index == 0:
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance index {}, return the first label.'.format(index))
            return self.instances[0]

    """返回字母表中实例的数量。如果字母表不是用于标签，那么还要加上默认索引（0）。"""
    def size(self):
        # if self.label:
        #     return len(self.instances)
        # else:
        return len(self.instances) + 1

    def iteritems(self):  #返回字典中的键值对
        return self.instance2index.items()

    """返回列表中从指定位置开始的索引和实例对。如果指定位置不在合法范围内，那么抛出异常。"""
    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False  #关闭字母表的增长功能

    def open(self):
        self.keep_growing = True

    def get_content(self):  #返回字母表中的内容，包括列表和字典。
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def _is_word_in_alphabet(self, word):
        # check if the word is in the alphabet
        if word in self.instance2index:
            return True
        else:
            return False
    def from_json(self, data):  #从json格式的数据中加载字母表的内容，并更新列表和字典。
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
