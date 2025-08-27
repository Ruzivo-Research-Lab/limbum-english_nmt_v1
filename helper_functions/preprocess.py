'''
This is a helper function to preprocess the data before training
'''
import re
import string
import unicodedata


# Converts the unicode file to ascii

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalize_diacritics_text(text_string):
    """Convenience wrapper to abstract away unicode & NFC"""
    return unicodedata.normalize("NFC", text_string)

# Modified to handle diacritics


def preprocess_sentence(w):
    w = normalize_diacritics_text(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    w = re_punc.sub('', w)

    lines_str = w.replace("”", "")
    lines_str = lines_str.replace("“", "")
    lines_str = lines_str.replace("’", "'")
    lines_str = lines_str.replace("«", "")
    lines_str = lines_str.replace("»", "")
    lines_str = ' '.join(
        [word for word in lines_str.split() if word.isalpha()])
    w = '<start> ' + lines_str + ' <end>'
    return w


def preprocess_sentence_1(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    w = re_punc.sub('', w)

    lines_str = w.replace("”", "")
    lines_str = lines_str.replace("“", "")
    lines_str = lines_str.replace("’", "'")
    lines_str = lines_str.replace("«", "")
    lines_str = lines_str.replace("»", "")
    lines_str = ' '.join(
        [word for word in lines_str.split() if word.isalpha()])
    return lines_str


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples] if
                  len(l.split("\t")) == 2 and preprocess_sentence_1(
                      l.split("\t")[1].strip("\n")) != "" and preprocess_sentence_1(
                      # to make sure the element has two pairs :
                      l.split("\t")[0].strip("\n")) != ""]
    # Limbum sentence and its English translation
    return zip(*word_pairs)


# en for Limbum, sp for English (names only used for prompt)
lmb, en = create_dataset(path_to_file, None)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    if src_lang.lower().strip() == "limbum":
        inp_lang, targ_lang = create_dataset(path, num_examples)
        # save_list(inp_lang, "training_limbum_sentences.txt")
        # save_list(targ_lang, "training_english_sentences.txt")
    else:
        targ_lang, inp_lang = create_dataset(path, num_examples)
        # not handled yet : This part will create the model for English - Limbum translation
        # save_list(inp_lang, "training_limbum_sentences.txt")
        # save_list(targ_lang, "training_english_sentences.txt")

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
