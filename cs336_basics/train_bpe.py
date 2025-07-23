
import regex as re
import os
import time
import cProfile
import pstats
import heapq

from collections import Counter
from typing import BinaryIO, Dict, List, Set, Tuple, Iterable, Iterator
from tqdm import tqdm

_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class PretokenWord:
  count = 0
  word = ""
  encoded_token_ids = []

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Build pretoken_words
    pretoken_words = {}
    with open(input_path, 'r') as file:
      for line_num, line in enumerate(file, 1):
        for break_line in re.split("|".join(special_tokens), line):
          for pretoken_word in re.findall(_PAT, break_line):
            if pretoken_word in pretoken_words:
              pretoken_words[pretoken_word].count += 1
            else:
              item = PretokenWord()
              item.word = pretoken_word
              item.count = 1
              utf8_encoded = pretoken_word.encode("utf-8")
              item.encoded_token_ids = [x + len(special_tokens) for x in list(utf8_encoded)]
              pretoken_words[pretoken_word] = item

    # Init vocab and merges
    vocab = {}
    merges = []
    for index, item in enumerate(special_tokens):
      vocab[index] = bytes(item, "utf-8")
    for i in range(256):
      vocab[i + len(special_tokens)] = bytes([i])
 
    # Merge tokens
    next_token_id = len(special_tokens) + 256
    for token_id in range(next_token_id, vocab_size): 
      # get frequency
      frequency = {}
      for key, val in pretoken_words.items():
        for i in range(len(val.encoded_token_ids) - 1):
          c_token_id = val.encoded_token_ids[i]
          c_1_token_id = val.encoded_token_ids[i + 1]
          if (c_token_id, c_1_token_id) not in frequency:
            frequency[(c_token_id, c_1_token_id)] = val.count
          else:
            frequency[(c_token_id, c_1_token_id)] += val.count

      # Order and update vocab and merges 
      if len(frequency) == 0:
        break
      max_frequency = max(frequency.values())
      max_keys_in_bytes = [(vocab[key[0]], vocab[key[1]], key[0], key[1]) for key, value in frequency.items() if value == max_frequency]
      max_key_in_bytes = max(max_keys_in_bytes)
      vocab[token_id] = max_key_in_bytes[0] + max_key_in_bytes[1]
      merges.append((max_key_in_bytes[0], max_key_in_bytes[1]))

      # Update pretoken_words
      updated = False
      for key, val in pretoken_words.items():
        lenth = len(val.encoded_token_ids)
        for i in range(lenth - 1, 0, -1):
          c_token_id = val.encoded_token_ids[i - 1]
          c_1_token_id = val.encoded_token_ids[i]
          if c_token_id == max_key_in_bytes[2] and c_1_token_id == max_key_in_bytes[3]:
            val.encoded_token_ids[i - 1] = token_id
            del val.encoded_token_ids[i]
            updated = True
      assert updated

    return vocab, merges


class PreTokenizer:
    def __init__(self, special_tokens: list[str]):
        """
        初始化预处理器
        
        Args:
            special_tokens: 特殊token列表
        """
        # 按长度降序排列特殊token，确保先匹配最长的token
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else r"(?!)" # 当没有特殊token时，使用永远不分割的表达式
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def find_chunk_boundaries(
        self,
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def read_corpus(self, input_path: str) -> Iterable[List[str]]:
        """读取语料并按特殊token进行分割"""
        with open(input_path, 'rb') as f:
            # 获取分块边界
            boundaries = self.find_chunk_boundaries(f, 100, "<|endoftext|>".encode('utf-8'))
        
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])), desc="读取语料"):
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
                yield re.split(self.special_tokens_pattern, chunk)
    
    def build_word_frequency(self, docs: Iterable[str]) -> Counter:
        """构建词频字典"""
        word_freq = Counter()
        str_freq = Counter()
        
        for doc in docs:
            if not doc:
                # 跳过空字符串
                continue
            # 收集doc中所有单词
            matches = [word.group(0) for word in self.word_pattern.finditer(doc)]
            # 批量更新计数器
            str_freq.update(matches)

        
        # 将字符串词频转换为字节词频
        for word, freq in str_freq.items():
            word_freq[word.encode('utf-8')] = freq

        return word_freq
    
    def pretokenize(self, text: str) -> List[bytes]:
        """
        将输入文本预分词，包括常规token和特殊token
        
        Args:
            text: 输入文本
        
        Returns:
            预分词结果bytes列表
        """
        # 首先按特殊token进行分割，但保留特殊token
        parts = re.split(f'({self.special_tokens_pattern})', text)
        
        result = []
        
        for part in parts:
            if part in self.special_tokens:
                # 特殊token直接放入
                result.append(part.encode('utf-8'))
            elif part:  # 跳过空字符串
                # 对普通文本使用word_pattern进行分词
                tokens = [match.group(0).encode('utf-8') for match in self.word_pattern.finditer(part)]
                result.extend(tokens)
        
        return result
    
    def pretokenize_iter(self, texts: Iterable[str]) -> Iterator[bytes]:
        """
        将输入的可迭代字符串对象预分词，并以迭代器形式返回词元
        
        Args:
            texts: 可迭代的字符串对象（例如文件句柄或字符串列表）
        
        Returns:
            生成预分词结果bytes的迭代器
        """
        for text in texts:
            # 首先按特殊token进行分割，但保留特殊token
            parts = re.split(f'({self.special_tokens_pattern})', text)
            
            for part in parts:
                if part in self.special_tokens:
                    # 特殊token直接生成
                    yield part.encode('utf-8')
                elif part:  # 跳过空字符串
                    # 对普通文本使用word_pattern进行分词并生成
                    for match in self.word_pattern.finditer(part):
                        yield match.group(0).encode('utf-8')


class CopyBPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        """
        初始化BPE分词器
        
        Args:
            vocab_size: 词汇表大小
            special_tokens: 特殊token列表
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.preprocessor = PreTokenizer(special_tokens)
        self.token_vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.splits: Dict[bytes, List[bytes]] = {}  # b"going" -> [b'g', b'o', b'ing'] 可以以此知道当前word有哪些pair
        self.pair_freqs: Dict[Tuple[bytes, bytes], int] = {}

        # 反向索引，记录每个pair出现在哪些单词中
        self.pair_to_words: Dict[Tuple[bytes, bytes], Set[bytes]] = {}
        # 最大堆，用于快速在pair_freqs中找到频率最高的pair
        self.freq_max_heap = []

    def _push_pair_to_heap(self, pair: Tuple[bytes, bytes], freq: int) -> None:
        # def bytes_to_lex_int(b: bytes) -> int:
        #     return int.from_bytes(b, byteorder='big')
        # inverted_pair = (-bytes_to_lex_int(pair[0]), -bytes_to_lex_int(pair[1]))
        # TODO：这里freq时排序有问题，插入不符合字典序
        heapq.heappush(self.freq_max_heap, (-freq, pair))
    
    def _pop_pair_from_heap(self) -> Tuple[bytes, bytes]:
        """从最大堆中弹出频率最高的字节对"""
        while self.freq_max_heap:
            freq, pair = heapq.heappop(self.freq_max_heap)
            freq = -freq
            if pair in self.pair_freqs and self.pair_freqs[pair] == freq:
                # 因为pair_freqs删除pair/减少某个pair的freq后，最大堆不立刻同步更新（使用懒惰删除策略）,所以要在弹出时进行检测
                # 如果不一样说明对应频率已经被减小/被删除
                return pair
        raise ValueError("堆没有返回频率最大的字节对")
    
    def initialize_splits_and_pairs(self, word_freqs: Counter) -> None:
        """初始化splits、pair_freqs、pair_to_words、freq_max_heap"""
        for word, word_freq in word_freqs.items():
            # 初始化splits，将单词转换为字节序列
            self.splits[word] = [bytes([b]) for b in word]

            word_pieces = self.splits[word]
            if len(word_pieces) == 1:
                continue
            for j, pair in enumerate(zip(word_pieces[:-1], word_pieces[1:])):
                # 扫描每个单词的每个字节对，初始化pair_freqs
                self.pair_freqs[pair] = self.pair_freqs.get(pair, 0) + word_freq

                # 记录pair出现在哪些单词中，初始化反向索引pair_to_words
                if pair not in self.pair_to_words:
                    self.pair_to_words[pair] = set()
                self.pair_to_words[pair].add(word)
        
        # 初始化最大堆
        for pair, freq in self.pair_freqs.items():
            self._push_pair_to_heap(pair, freq)
        
    
    def find_best_pair(self) -> Tuple[bytes, bytes]:
        """找到频率最高的字节对"""
        return self._pop_pair_from_heap()
    
    def _update_pair_freqs(self, new_pair, old_pair, word, word_freq) -> None:
        # 添加 new_pair
        self.pair_to_words.setdefault(new_pair, set()).add(word)
        self.pair_freqs[new_pair] = self.pair_freqs.get(new_pair, 0) + word_freq
        # 一个new_pair可能被多次添加到堆中，但是应该问题不大
        self._push_pair_to_heap(new_pair, self.pair_freqs[new_pair])

        # 减少 old_pair
        if old_pair in self.pair_freqs:
            self.pair_freqs[old_pair] -= word_freq
            if self.pair_freqs[old_pair] <= 0:
                del self.pair_freqs[old_pair]
            else:
                # 如果old_pair仍然存在，更新最大堆
                self._push_pair_to_heap(old_pair, self.pair_freqs[old_pair])
        # 这里就不删除pair_to_words和freq_max_heap的对应项了，前者我们只关心我们要查的pair有就行，后者我们在取出时会检查
    
    def update_splits_and_pairs(self, best_pair: Tuple[bytes, bytes], new_token: bytes, word_freqs: Counter) -> None:
        """更新splits和pair_freqs"""
        # 哪些词包含best_pair，需要被更新
        # 直接从反向索引中获取
        affected_words = list(self.pair_to_words.get(best_pair, set()))

        # 更新splits
        for word in affected_words:
            word_freq = word_freqs[word]
            word_pieces = self.splits[word]
            i = 0
            while i < len(word_pieces) - 1:
                if word_pieces[i] == best_pair[0] and word_pieces[i + 1] == best_pair[1]:
                    # 如果找到best_pair，合并
                    word_pieces[i] = new_token
                    word_pieces.pop(i + 1)

                    # 删除best_pair在pair_freqs中的记录
                    if best_pair in self.pair_freqs:
                        del self.pair_freqs[best_pair]

                    # 如果合并后左侧或右侧还有元素，则会出现新对，影响旧对
                    # 要更新pair_freqs、pair_to_words
                    # 假设合并前word_pieces为 [A, B, C, D]，合并后变为 [A, BC, D]
                    if i > 0:
                        # 添加 A, BC ; 减少 A, B
                        new_pair_left = (word_pieces[i-1], new_token)
                        old_pair_left = (word_pieces[i-1], best_pair[0])
                        self._update_pair_freqs(new_pair_left, old_pair_left, word, word_freq)
                    if i < len(word_pieces) - 1:
                        # 添加 BC, D ; 减少 C, D
                        new_pair_right = (new_token, word_pieces[i+1])
                        old_pair_right = (best_pair[1], word_pieces[i+1])
                        self._update_pair_freqs(new_pair_right, old_pair_right, word, word_freq)
                else:
                    i += 1
        
    def add_special_tokens(self) -> None:
        """将特殊token添加到词汇表中"""
        for m, token in enumerate(self.special_tokens):
            self.token_vocab[self.vocab_size - len(self.special_tokens) + m] = token.encode('utf-8')
    
    def train(self, input_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """训练BPE模型"""
        # 读取语料
        for docs in self.preprocessor.read_corpus(input_path):
            # 预分词，建立词频字典
            word_freq = self.preprocessor.build_word_frequency(docs)
        
        # BPE训练开始
        
        # 初始化词汇表和合并表
        self.token_vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        self.merges = []
        
        # 初始化splits和pair_freqs
        self.initialize_splits_and_pairs(word_freq)
        
        # 执行合并
        for num_merge in range(num_merges):
            if not self.pair_freqs:
                break
            
            # 选择频率最高的pair
            best_pair = self.find_best_pair()
            self.merges.append(best_pair)
            
            # 更新词汇表，添加新的合并token
            new_token = best_pair[0] + best_pair[1]
            self.token_vocab[256 + num_merge] = new_token
            
            # 更新splits和pair_freqs
            self.update_splits_and_pairs(best_pair, new_token, word_freq)
        
        # 添加特殊token到词汇表
        self.add_special_tokens()
        
        return self.token_vocab, self.merges
    
    def to_files(self, vocab_filepath: str, merges_filepath: str) -> None:
        """将训练结果保存到文件"""
        # 保存词汇表
        with open(vocab_filepath, 'wb') as f:
            # 写入词汇表大小
            f.write(len(self.token_vocab).to_bytes(4, byteorder='little'))
            
            # 写入每个token: <id(4字节)><长度(4字节)><token内容(bytes)>
            for token_id, token in self.token_vocab.items():
                f.write(token_id.to_bytes(4, byteorder='little'))
                f.write(len(token).to_bytes(4, byteorder='little'))
                f.write(token)
        
        # 保存合并规则
        with open(merges_filepath, 'wb') as f:
            # 写入合并规则数量
            f.write(len(self.merges).to_bytes(4, byteorder='little'))
            
            # 写入每个合并规则: <第一部分长度(4字节)><第一部分内容(bytes)><第二部分长度(4字节)><第二部分内容(bytes)>
            for first, second in self.merges:
                f.write(len(first).to_bytes(4, byteorder='little'))
                f.write(first)
                f.write(len(second).to_bytes(4, byteorder='little'))
                f.write(second)
        

def copy_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    tokenizer = CopyBPETrainer(vocab_size, special_tokens)
    return tokenizer.train(input_path)
