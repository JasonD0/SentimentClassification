import tensorflow as tf

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    processed_review = review

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    """
    Input placeholder: name="input_data"
    labels placeholder: name="labels"
    accuracy tensor: name="accuracy"
    loss tensor: name="loss"
    """

    input_data = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32, name="input_data")
    labels = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32, name="labels")
    dropout_keep_prob = tf.placeholder_with_default(dtype=tf.float32, name="dropout_keep_prob")

    # rnn layers
    lstm_cell = tf.nn.rnn_cell.LSTMCell(128)
    lstm_cell_dropped = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_dropped for _ in range(2)])
    rnn_ouputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=input_data, initial_state=lstm_cells.zero_state(BATCH_SIZE, dtype=tf.float32), dtype=tf.float32)

    # dense layer 
    w = tf.Variable(tf.random_normal([rnn_outputs.get_shape().as_list()[1], 2], stddev=0.1), dtype=tf.float32, name="weights")
    b = tf.Variable(tf.zeros([2]), dtype=tf.float32, name="biases")
    preds = tf.nn.softmax(tf.matmul(rnn_outputs, w) + b)
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(preds + 1e-7)), name="loss")

    Accuracy = tf.reduce_mean(tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1)), dtype=tf.float32, name="accuracy")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)#.minimise(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
