
import regex as re

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
