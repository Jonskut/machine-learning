import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
# print('Vocabulary size')
# print(len(vocab))
# print(vocab['man'])
# print(len(ivocab))
# print(ivocab[10])

# W contains vectors for
# print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v


def get_vector(prompt, vocabulary):
    while True:
        input_word = input(prompt)
        if input_word == "EXIT":
            return None
        if input_word in vocabulary:
            return input_word, W[vocabulary[input_word], :]
        else:
            print("Word not in vocabulary, try again.")


# Main loop for analogy
while True:
    print("Enter 3 words. Enter EXIT to exit")
    x = get_vector("Enter word X: ", vocab)
    if x is None:
        break
    word_x, vector_x = x

    y = get_vector("Enter word Y: ", vocab)
    if x is None:
        break
    word_y, vector_y = y

    z = get_vector("Enter word Z: ", vocab)
    if x is None:
        break
    word_z, vector_z = z

    # Analogy vector
    input_vector = vector_z + (vector_y - vector_x)

    # Search for nearest neighbor to analogy vector
    similar_words = []
    for word, idx in vocab.items():
        comparison_vector = W[idx, :]
        distance = \
            float(np.sqrt(np.sum((input_vector - comparison_vector) ** 2)))

        similar_words.append((word, distance))

    # Sort by distance
    similar_words.sort(key=lambda x: x[1])

    # Print out
    print(f"{word_x} is to {word_y} as {word_z} is to")
    print("\n                               Word       Distance\n")
    print("---------------------------------------------------------\n")
    for i in range(2):
        word, distance = similar_words[i]
        print("%35s\t\t%f\n" % (word, distance))
