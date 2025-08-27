from seaborn import load_dataset
from sklearn.model_selection import train_test_split

from helper_functions.preprocess import max_length


num_examples = int(0.9 * len(lmb))

print("Total Dataset Size : {} - Training Size : {} - Testing Size (with BLEU) : {}".format(len(lmb), num_examples,
                                                                                            len(lmb) - num_examples))
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(
    target_tensor), max_length(input_tensor)

# Creating training and validation sets using an 90-10 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.1)

# parameters chosen after many trials :)
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 100
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
units = 128
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1
embedding_dim = 512

print("Limbum vocabulary size : {} - English vocabulary : {}".format(vocab_inp_size, vocab_tar_size))

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
