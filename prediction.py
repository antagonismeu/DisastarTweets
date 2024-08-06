import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score


def configuration() :
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    

class TextClassificationModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, common_length):
        super(TextClassificationModel, self).__init__()
        self.keyword_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=common_length)
        self.location_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=common_length)
        self.text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=common_length)
        self.attention = Attention()
        self.attention2 = Attention()
        self.text_dense = Dense(common_length)
        self.keyword_dense = Dense(common_length)
        self.location_dense = Dense(common_length)
        self.gru1 = Bidirectional(GRU(64, return_sequences=True))
        self.gru2 = Bidirectional(GRU(32, return_sequences=True))
        self.flatten = Flatten()
        self.dense1 = Dense(16, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        keyword_input, location_input, text_input = inputs
        keyword_emb = self.keyword_embedding(keyword_input)
        location_emb = self.location_embedding(location_input)
        text_emb = self.text_embedding(text_input)

        keyword_emb = self.keyword_dense(tf.transpose(keyword_emb, perm=[0, 2, 1]))
        location_emb = self.location_dense(tf.transpose(location_emb, perm=[0, 2, 1]))
        text_emb = self.text_dense(tf.transpose(text_emb, perm=[0, 2, 1]))
        context_vector, attention_weights = self.attention([keyword_emb, location_emb, text_emb], return_attention_scores=True)
        context_vector = tf.transpose(context_vector, perm=[0, 2, 1])
        
        
        x = self.gru1(context_vector)
        x, _ = self.attention2([x, x, x], return_attention_scores=True)
        x = self.gru2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    def train_step(self, data):
        (keyword_sequences, location_sequences, text_sequences), targets = data
        with tf.GradientTape() as tape:
            predictions = self([keyword_sequences, location_sequences, text_sequences], training=True)
            loss = self.compiled_loss(targets, predictions, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}



class ModelEvaluator:
    def __init__(self, model, tokenizer, text_maxlen=100, keyword_maxlen=3, location_maxlen=20):
        self.model = model
        self.tokenizer = tokenizer
        self.text_maxlen = text_maxlen
        self.keyword_maxlen = keyword_maxlen
        self.location_maxlen = location_maxlen

    def preprocess_test_data(self, file_path):
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)

        texts = df['text'].tolist()
        keywords = df['keyword'].tolist()
        locations = df['location'].tolist()
        targets = df['target'].tolist()
        text_sequences = self.tokenizer.texts_to_sequences(texts)
        keyword_sequences = self.tokenizer.texts_to_sequences(keywords)
        location_sequences = self.tokenizer.texts_to_sequences(locations)
        text_sequences_padded = pad_sequences(text_sequences, maxlen=self.text_maxlen, padding='post')
        keyword_sequences_padded = pad_sequences(keyword_sequences, maxlen=self.keyword_maxlen, padding='post')
        location_sequences_padded = pad_sequences(location_sequences, maxlen=self.location_maxlen, padding='post')
        targets_array = np.array(targets)
        return (keyword_sequences_padded, location_sequences_padded, text_sequences_padded), targets_array

    def evaluate(self, test_file_path):
        test_data, true_labels = self.preprocess_test_data(test_file_path)
        predictions = self.model.predict(test_data)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        print(predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def generate_submission(self, test_file_path, output_file_path):
        df_test = pd.read_csv(test_file_path)
        df_test.fillna('', inplace=True)
        
        texts = df_test['text'].tolist()
        keywords = df_test['keyword'].tolist()
        locations = df_test['location'].tolist()
        ids = df_test['id'].tolist()  
        
        text_sequences = self.tokenizer.texts_to_sequences(texts)
        keyword_sequences = self.tokenizer.texts_to_sequences(keywords)
        location_sequences = self.tokenizer.texts_to_sequences(locations)
        
        text_sequences_padded = pad_sequences(text_sequences, maxlen=self.text_maxlen, padding='post')
        keyword_sequences_padded = pad_sequences(keyword_sequences, maxlen=self.keyword_maxlen, padding='post')
        location_sequences_padded = pad_sequences(location_sequences, maxlen=self.location_maxlen, padding='post')
        
        test_data = (keyword_sequences_padded, location_sequences_padded, text_sequences_padded)
        predictions = self.model.predict(test_data)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        submission_df = pd.DataFrame({
            'id': ids,
            'target': predicted_labels
        })
        submission_df.to_csv(output_file_path, index=False)

class TextClassificationSystem:
    def __init__(self, train_file_path, test_file_path, validation_file_path, output_file_path, embedding_dim=16, text_maxlen=100, keyword_maxlen=3, location_maxlen=20, common_dim=41):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.validation_file_path = validation_file_path
        self.output_file_path = output_file_path
        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen
        self.keyword_maxlen = keyword_maxlen
        self.location_maxlen = location_maxlen
        self.common_dim = common_dim
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.vocab_size = self._compute_vocab_size()
        self.model = TextClassificationModel(self.vocab_size, self.embedding_dim, self.common_dim)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.train_dataset, self.val_dataset, self.val_data_df = self._preprocess_data(self.train_file_path)
        self.evaluator = ModelEvaluator(self.model, self.tokenizer, self.text_maxlen, self.keyword_maxlen, self.location_maxlen)

    def _compute_vocab_size(self):
        df = pd.read_csv(self.train_file_path)
        df.fillna('', inplace=True)
        texts = df['text'].tolist()
        keywords = df['keyword'].tolist()
        locations = df['location'].tolist()
        all_texts = texts + keywords + locations
        self.tokenizer.fit_on_texts(all_texts)
        vocab_size = len(self.tokenizer.word_index) + 1
        return vocab_size

    def _preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        texts = df['text'].tolist()
        keywords = df['keyword'].tolist()
        locations = df['location'].tolist()
        targets = df['target'].tolist()
        
        text_sequences = self.tokenizer.texts_to_sequences(texts)
        keyword_sequences = self.tokenizer.texts_to_sequences(keywords)
        location_sequences = self.tokenizer.texts_to_sequences(locations)
        
        text_sequences_padded = pad_sequences(text_sequences, maxlen=self.text_maxlen, padding='post')
        keyword_sequences_padded = pad_sequences(keyword_sequences, maxlen=self.keyword_maxlen, padding='post')
        location_sequences_padded = pad_sequences(location_sequences, maxlen=self.location_maxlen, padding='post')
        
        targets_array = np.array(targets)
        dataset = tf.data.Dataset.from_tensor_slices(((keyword_sequences_padded, location_sequences_padded, text_sequences_padded), targets_array))
        dataset = dataset.shuffle(buffer_size=10000).batch(32).prefetch(tf.data.AUTOTUNE)
        val_size = int(0.3 * len(targets_array))  
        train_size = len(targets_array) - val_size
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        
        val_data_df = df.iloc[train_size:]
        val_data_df.to_csv(self.validation_file_path, index=False)

        return train_dataset, val_dataset, val_data_df

    def train(self, epochs=10):
        self.model.fit(self.train_dataset, epochs=epochs)
        self.model.summary()
        self.model.save('/kaggle/working/text_classification_model.h5')
    
    def evaluate(self):
        # Evaluate the model using the validation dataset and save the results to a CSV file
        metrics = self.evaluator.evaluate(self.validation_file_path)
        return metrics
    
    def generate_submission(self):
        self.evaluator.generate_submission(self.test_file_path, self.output_file_path)



def main() :
    configuration()
    train_file_path = '/kaggle/input/nlp-getting-started/train.csv'
    test_file_path = '/kaggle/input/nlp-getting-started/test.csv'
    validation_file_path = '/kaggle/working/validation_data.csv'
    output_file_path = '/kaggle/working/sample_submission.csv'
    system = TextClassificationSystem(train_file_path, test_file_path, validation_file_path, output_file_path)
    system.train(epochs=30)
    metrics = system.evaluate()
    print('Evaluation Metrics:', metrics)
    system.generate_submission()
    print('csv file has generated !')



if __name__ == '__main__' :
    main()


