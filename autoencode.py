import tensorflow as tf
import symbols
import numpy as np

# DATA_PATH = 'data/allsets.txt'
DATA_PATH = 'data/akh.txt'

MODEL_PATH = 'models/model.ckpt'

NUM_HIDDEN_ENCODER = 100
NUM_HIDDEN_DECODER = 100
FEATURE_BITS = 12
NUM_LAYERS_ENCODER = 2
NUM_LAYERS_DECODER = 2
BATCH_SIZE = 50

if __name__ == '__main__':
    card_length = 0
    cards = None
    symbol_table = None
    with open(DATA_PATH, 'r') as f:
        data = f.read()
        symbol_table = symbols.create_symbol_table(data)
        card_strings = data.split('\n\n')[:-1]
        card_length = symbols.get_max_card_length(card_strings)
        cards, card_lengths = symbols.card_array_as_symbol_array(symbol_table, card_strings, card_length)

    char_length = len(symbol_table.keys())
    start_symbol = tf.reshape(tf.tile(tf.constant(symbol_table['|'], dtype=tf.float32), [BATCH_SIZE]), [BATCH_SIZE, char_length])

    cards_in = tf.placeholder(tf.float32, [None, card_length, char_length])
    cards_valid_indices = tf.placeholder(tf.float32, [None, card_length])
    card_length_in = tf.placeholder(tf.int32, [BATCH_SIZE])

    fc_weights = {
        'w_encoder': tf.Variable(tf.random_normal([NUM_HIDDEN_ENCODER, FEATURE_BITS], stddev=1.0/np.sqrt(NUM_HIDDEN_ENCODER))),
        'b_encoder': tf.Variable(tf.random_normal([FEATURE_BITS], stddev=1.0/np.sqrt(NUM_HIDDEN_DECODER))),
        'w_decoder': tf.Variable(tf.random_normal([NUM_HIDDEN_DECODER, char_length], stddev=1.0/np.sqrt(NUM_HIDDEN_DECODER))),
        'b_decoder': tf.Variable(tf.random_normal([char_length], stddev=1.0/np.sqrt(NUM_HIDDEN_DECODER))),
    }

    # lstm
    cards_split = tf.reshape(cards_in, [-1, card_length, char_length])
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_ENCODER) for _ in range(NUM_LAYERS_ENCODER)])
    lstm_outs, lstm_states = tf.nn.dynamic_rnn(stacked_lstm, cards_split, dtype=tf.float32, sequence_length=card_length_in)

    # get last output
    last_output = tf.gather_nd(lstm_outs, tf.stack([tf.range(BATCH_SIZE), card_length_in-1], axis=1))
    card_vector = tf.nn.relu(tf.add(tf.matmul(tf.reshape(last_output, [-1, NUM_HIDDEN_ENCODER]), fc_weights['w_encoder']), fc_weights['b_encoder']))

    # run rnn in reverse here, building both a network for training and one for actual decoding
    # single character stack w/ last character and card vector as input
    single_lstm = None
    with tf.variable_scope('decoder'):
        single_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_DECODER) for _ in range(NUM_LAYERS_DECODER)])

    decoder_unfurled_gen = [(start_symbol, None)]
    decoder_unfurled_train = [(start_symbol, None)]
    for i in range(card_length-1):

        with tf.variable_scope('decoder') as scope:
            print i
            if i > 0:
                scope.reuse_variables()

            # network for training is fed the full card, and compares the output for every card
            feed_character = tf.gather_nd(cards_split, tf.stack([tf.range(BATCH_SIZE), tf.ones([BATCH_SIZE], dtype=tf.int32)*i], axis=1))
            in_data_train = tf.concat([card_vector, feed_character], 1)
            init_state_train = decoder_unfurled_train[-1][1] if decoder_unfurled_train[-1][1] is not None else single_lstm.zero_state(BATCH_SIZE, tf.float32)
            iteration_outs_train, iteration_states_train = single_lstm(in_data_train, init_state_train)
            # get state and connect to fc layer with a softmax later
            character_train = tf.nn.relu(tf.add(tf.matmul(tf.reshape(iteration_outs_train, [-1, NUM_HIDDEN_DECODER]), fc_weights['w_decoder']), fc_weights['b_decoder']))
            decoder_unfurled_train.append((character_train, iteration_states_train))

            # network for generation recursively generates cards
            scope.reuse_variables()
            in_data_gen = tf.concat([card_vector, decoder_unfurled_gen[-1][0]], 1)
            init_state_gen = decoder_unfurled_gen[-1][1] if decoder_unfurled_gen[-1][1] is not None else single_lstm.zero_state(BATCH_SIZE, tf.float32)
            iteration_outs_gen, iteration_states_gen = single_lstm(in_data_gen, init_state_gen)
            character_gen = tf.nn.relu(tf.add(tf.matmul(tf.reshape(iteration_outs_gen, [-1, NUM_HIDDEN_DECODER]), fc_weights['w_decoder']), fc_weights['b_decoder']))
            decoder_unfurled_gen.append((character_gen, iteration_states_gen))

    output_train = [tf.reshape(h[0], [BATCH_SIZE, 1, char_length]) for h in decoder_unfurled_train]
    output_train = tf.concat(output_train, 1)

    cost = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=cards_split, logits=output_train), cards_valid_indices))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    output_gen = [tf.reshape(h[0], [BATCH_SIZE, 1, char_length]) for h in decoder_unfurled_gen]
    output_gen = tf.concat(output_gen, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver.restore(sess, MODEL_PATH)
            print "Loaded previous run!"
        except:
            print "no saved model, starting anew"

        for i in range(100000):
            if i % 1000 == 0:
                print "saving"
                saver.save(sess, MODEL_PATH)

            batch, batch_lengths, batch_indices = symbols.get_batch(BATCH_SIZE, cards, card_lengths)
            cost_eval = sess.run(cost, feed_dict={cards_in: batch, card_length_in: batch_lengths, cards_valid_indices: batch_indices})
            cards_fed = sess.run(cards_split, feed_dict={cards_in: batch, card_length_in: batch_lengths, cards_valid_indices: batch_indices})
            cards_out = sess.run(output_train, feed_dict={cards_in: batch, card_length_in: batch_lengths, cards_valid_indices: batch_indices})
            print "epoch %d, cost: %0.5f" % (i, cost_eval)
            print symbols.symbol_array_as_card_array(symbol_table, [cards_fed[0]])
            print symbols.symbol_array_as_card_array(symbol_table, [cards_out[0]])
            sess.run(optimizer, feed_dict={cards_in: batch, card_length_in: batch_lengths, cards_valid_indices: batch_indices})
