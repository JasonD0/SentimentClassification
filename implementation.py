import tensorflow as tf
import re 

BATCH_SIZE = 350
MAX_WORDS_IN_REVIEW = 220  # Maximum length of a review to consider
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
                  'how', 'further', 'was', 'here', 'than', 'would'})

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
    
    processed_review = review.lower()
    for stopWord in stop_words:
        stopWord = "\b" + stopWord + "\b"
        processed_review = re.sub(stopWord, '', processed_review)
    processed_review = re.sub('<br />', '', processed_review)
    processed_review = re.sub('(\'s|\'re|\'ve)', '', processed_review)
    processed_review = re.sub('\s+', ' ', processed_review)
    processed_review = re.sub('[_~`!@#$%^&\*\(\)\{\}\[\]:;\"\'\?/>\.<,=-]', '', processed_review)
    processed_review = processed_review.replace('\\', '')
    processed_review = processed_review.replace('+', '')
    
    rev = processed_review.split()
    return rev



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
    tf.reset_default_graph()

    input_data = tf.placeholder(shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], dtype=tf.float32, name="input_data")
    labels = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="labels")
    dropout_keep_prob = tf.placeholder_with_default(0.5, shape=(), name="dropout_keep_prob")

    #input_data = tf.nn.dropout(input_data, dropout_keep_prob)

    #conv layer with max pooling
    conv = tf.layers.conv1d(inputs=input_data, filters=8, kernel_size=2, strides=2, padding="same", activation=tf.nn.relu)
    pooling = tf.layers.max_pooling1d(inputs=conv, pool_size=2, padding="same", strides=2)
    #pooling = tf.nn.dropout(pooling, dropout_keep_prob)

    conv = tf.layers.conv1d(inputs=pooling, filters=4, kernel_size=2, strides=2, padding="same", activation=tf.nn.relu)
    pooling = tf.layers.max_pooling1d(inputs=conv, pool_size=2, padding="same", strides=2)
    #pooling = tf.nn.dropout(pooling, dropout_keep_prob)

    # rnn layers
    lstm_cells = []
    for _ in range(1):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(32)  
        lstm_cell_dropped = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
        lstm_cells.append(lstm_cell_dropped)

    multi_cells = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=multi_cells, inputs=pooling, initial_state=multi_cells.zero_state(BATCH_SIZE, dtype=tf.float32), dtype=tf.float32)
    rnn_outputs = rnn_outputs[:,-1]

    # dense layer     
    w = tf.Variable(tf.truncated_normal([rnn_outputs.get_shape().as_list()[1], 16], stddev=0.1), dtype=tf.float32, name="weights")
    b = tf.Variable(tf.zeros([16]), dtype=tf.float32, name="biases")

    w1 = tf.Variable(tf.truncated_normal([16, 2], stddev=0.1), dtype=tf.float32, name="weights")
    b1 = tf.Variable(tf.zeros([2]), dtype=tf.float32, name="biases")

    preds = tf.nn.relu(tf.matmul(rnn_outputs, w) + b)
    preds = tf.nn.dropout(preds, dropout_keep_prob)

    preds = tf.nn.softmax(tf.matmul(preds, w1) + b1)
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(preds + 1e-2)), name="loss")

    # weight decay
    regulariser = tf.nn.l2_loss(w) + tf.nn.l2_loss(w1)
    loss = tf.reduce_mean(loss + 0.0001*regulariser)

    Accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds,1), tf.argmax(labels,1)), tf.float32), name="accuracy")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss