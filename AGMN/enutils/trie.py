import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):   #插入函数，用于将一个单词（word）添加到字典树中
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):  #搜索函数，用于判断一个单词（word）是否存在于字典树中。
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):   #前缀匹配函数，用于判断一个前缀（prefix）是否存在于字典树中
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerateMatch(self, word, space="_", backward=False):  #space=‘’   #枚举匹配函数，用于找出一个单词（word）在字典树中所有匹配的后缀，并以列表（matched）的形式返回
        matched = []

        while len(word) > 0:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

