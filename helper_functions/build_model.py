
'''
Class for building the model- Unidirectional GRU Encoder-Decoder with Bahdanau Attention
'''
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Plain GRU via GRUCell inside RNN (unidirectional)
        self.gru = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.enc_units),
            return_sequences=True,
            return_state=True
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        # RNN(..., return_state=True) **always** returns (outputs, state)
        outputs, state = self.gru(x, initial_state=[hidden])
        return outputs, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print(sample_output.shape, sample_hidden.shape)  # (B, T, units), (B, units)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) +
                       self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)  # (B, T, 1)
        context_vector = attention_weights * values       # (B, T, U)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (B, U)
        return context_vector, attention_weights


# (sanity check)
attention_layer = BahdanauAttention(30)
attention_result, attention_weights = attention_layer(
    sample_hidden, sample_output)


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Use RNN(GRUCell) to guarantee (outputs, state) in TF2
        self.gru = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.dec_units),
            return_sequences=True,
            return_state=True
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Bahdanau attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # Attention
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # Embed current token ids -> (B, 1, emb)
        x = self.embedding(x)

        # Concat context + embedding -> (B, 1, emb + dec_units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # NOTE: Fon didn't pass initial_state to the decoder GRU; keep that behavior
        # returns (sequence, state) in TF2 with RNN(GRUCell)
        outputs, state = self.gru(x)

        # Flatten time dimension (length 1) -> (B, dec_units)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        # Project to vocab -> (B, vocab_size)
        logits = self.fc(outputs)

        return logits, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# IMPORTANT: token ids must be int32
test_tokens = tf.random.uniform(
    (BATCH_SIZE, 1), maxval=vocab_tar_size, dtype=tf.int32)

sample_decoder_output, dec_state, attn_w = decoder(
    test_tokens,       # (B, 1) int32 ids
    sample_hidden,     # (B, units) from encoder
    sample_output      # (B, T, units) encoder outputs
)

print("logits:", sample_decoder_output.shape)  # (BATCH_SIZE, vocab_tar_size)
print("state :", dec_state.shape)              # (BATCH_SIZE, units)
print("attn  :", attn_w.shape)                 # (BATCH_SIZE, T, 1)
