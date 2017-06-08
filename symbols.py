import numpy as np
import random

# create symbol table, make "stop symbol" [0, 0 .... 1], aka ';'
def create_symbol_table(all_data):
    table = {}
    index = 0
    for char in all_data:
        if char not in table.keys():
            table[char] = index
            index += 1
    table[';'] = index

    # replace with numpy arrays
    array_table = {}
    base_array = np.zeros([len(table.keys())])
    for key, value in table.iteritems():
        this_array = base_array.copy()
        this_array[value] = 1.0
        array_table[key] = this_array
    return array_table

def get_max_card_length(cards):
    max_length = -1
    for card in cards:
        if len(card) > max_length:
            max_length = len(card)
    return max_length

# convert card to symbol array, adding stop symbol to the end
def card_as_symbol_array(table, convertee, length):
    converted = []
    card_length = 0
    for char in convertee:
        converted.append(table[char])
        card_length += 1
    while len(converted) < length:
        converted.append(table[';'])
    return np.array(converted), card_length

def card_array_as_symbol_array(table, convertee_array, length):
    converted_array = []
    converted_length_array = []
    for convertee in convertee_array:
        card_data, card_length = card_as_symbol_array(table, convertee, length)
        converted_array.append(card_data)
        converted_length_array.append(card_length)
    return np.array(converted_array), np.array(converted_length_array)

def symbol_array_as_card_array(table, convertee_array):
    cards = []
    keys = table.keys()
    values = table.values()
    for convertee in convertee_array:
        card = ''
        for symbol in convertee:
            index = 0
            # TODO do a better search
            for i, v in enumerate(values):
                max_array = np.zeros_like(symbol)
                max_array[np.argmax(symbol)] = 1.0
                if np.dot(v, max_array) > 0:
                    index = i
                    break
            card += keys[index]
        cards.append(card)
    return cards

def get_batch(batch_size, card_data, card_lengths):
    max_length = card_data.shape[0]
    batch = np.zeros([batch_size] + list(card_data.shape[1:]))
    batch_indices = np.zeros([batch_size, card_data.shape[1]])
    batch_length = np.zeros([batch_size])
    indices = random.sample(range(1, max_length-1), batch_size)
    for i in range(batch_size):
        index = indices[i]
        batch[i,:,:] = card_data[index]
        batch_length[i] = card_lengths[index]
        batch_indices[i,:card_lengths[index]] = 1.0
    return batch, batch_length, batch_indices