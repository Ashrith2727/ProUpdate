import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Concatenate, Attention 
from nltk.corpus import reuters 
import nltk 
import numpy as np 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu 
# Download the Reuters dataset 
nltk.download('Xsum') 
# Load the Reuters dataset 
docs = reuters.fileids() 
train_docs_id = list(filter(lambda doc: doc.startswith("train"), docs)) 
test_docs_id = list(filter(lambda doc: doc.startswith("test"), docs)) 
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id] 
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id] 
# Tokenize and preprocess the text data 
max_words = 10000 
max_len = 100  # Adjust as needed 
tokenizer = Tokenizer(num_words=max_words) 
tokenizer.fit_on_texts(train_docs) 
word_index = tokenizer.word_index 
X_train_seq = tokenizer.texts_to_sequences(train_docs) 
X_test_seq = tokenizer.texts_to_sequences(test_docs) 
X_train = pad_sequences(X_train_seq, maxlen=max_len, padding='post') 
X_test = pad_sequences(X_test_seq, maxlen=max_len, padding='post') 
# Create target summaries 
Y_train = tokenizer.texts_to_sequences([reuters.raw(doc_id).split('\n')[0] for doc_id in train_docs_id]) 
Y_test = tokenizer.texts_to_sequences([reuters.raw(doc_id).split('\n')[0] for doc_id in test_docs_id]) 
# Ensure that sequences have the same length as input sequences 
Y_train = pad_sequences(Y_train, maxlen=max_len, padding='post') 
Y_test = pad_sequences(Y_test, maxlen=max_len, padding='post') 
# Encoder 
encoder_inputs = Input(shape=(max_len,)) 
embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=200, input_length=max_len) 
embedded_inputs = embedding_layer(encoder_inputs) 
31 
# Increase the complexity of the encoder with more Bidirectional LSTM layers 
encoder_lstm_1 = Bidirectional(LSTM(256, return_sequences=True))(embedded_inputs) 
encoder_lstm_2 = Bidirectional(LSTM(128, return_sequences=True))(encoder_lstm_1) 
encoder_lstm_3 = Bidirectional(LSTM(64, return_sequences=True))(encoder_lstm_2) 
# Attention Mechanism 
attention = Attention()([encoder_lstm_3, encoder_lstm_2]) 
context = tf.reduce_sum(attention * encoder_lstm_2, axis=1, keepdims=True) 
decoder_inputs = Input(shape=(max_len,)) 
decoder_embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=200, input_length=max_len) 
decoder_embedded_inputs = decoder_embedding_layer(decoder_inputs) 
# Concatenate attention context with decoder inputs 
decoder_lstm_input = Concatenate(axis=-1)([decoder_embedded_inputs, context]) 
# Increase the complexity of the decoder with more LSTM layers 
decoder_lstm_1 = LSTM(64, return_sequences=True) 
decoder_lstm_2 = LSTM(128, return_sequences=True) 
decoder_lstm_3 = LSTM(256, return_sequences=True) 
decoder_lstm_1_out = decoder_lstm_1(decoder_lstm_input) 
decoder_lstm_2_out = decoder_lstm_2(decoder_lstm_1_out) 
decoder_lstm_3_out = decoder_lstm_3(decoder_lstm_2_out) 
# Concatenate decoder LSTM output and attention context 
decoder_combined_context = Concatenate(axis=-1)([decoder_lstm_3_out, context]) 
# Add more Dense layers in the decoder for better representation 
decoder_dense_1 = Dense(128, activation='relu') 
decoder_dense_2 = Dense(len(word_index) + 1, activation='softmax') 
decoder_dense_1_out = decoder_dense_1(decoder_combined_context) 
decoder_outputs = decoder_dense_2(decoder_dense_1_out) 
# Model 
seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
seq2seq_Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
# Training 
seq2seq_Model.fit([X_train, Y_train], Y_train, epochs=4, batch_size=64, validation_split=0.2) 
seq2seq_Model.save('G:/TextSummModel/models/modelsseq2seq_model.h5') 
# Evaluation 
loss, accuracy = seq2seq_Model.evaluate([X_test, Y_test], Y_test) 
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}') 
seq2seq_Model.summary() 
# Generate summaries using the trained model 
try: 
predicted_summaries = seq2seq_Model.predict([X_test, Y_test]) 
except Exception as e: 
print(f"Error during prediction: {e}") 
 
predicted_summaries_text = [] 
ground_truth_summaries_text = [] 
 
try: 
    for summary in predicted_summaries: 
        predicted_summaries_text.append(tokenizer.sequences_to_texts([np.argmax(token) for token in summary])) 
except Exception as e: 
    print(f"Error converting predicted summaries to text: {e}") 
 
try: 
    for summary in Y_test: 
        ground_truth_summaries_text.append(tokenizer.sequences_to_texts([token for token in summary])) 
except Exception as e: 
    print(f"Error converting ground truth summaries to text: {e}") 
 
# Calculate BLEU score for each example 
bleu_scores = [] 
for ref, hyp in zip(ground_truth_summaries_text, predicted_summaries_text): 
    try: 
        bleu_scores.append(sentence_bleu([ref.split()], hyp.split())) 
    except Exception as e: 
        print(f"Error calculating BLEU score for an example: {e}") 
 
# Calculate the overall BLEU score for the entire test set 
try: 
    overall_bleu_score = corpus_bleu([[ref.split()] for ref in ground_truth_summaries_text], 
predicted_summaries_text) 
except Exception as e: 
    print(f"Error calculating overall BLEU score: {e}") 
 
print(f'Overall BLEU Score: {overall_bleu_score}') 
 
# Optionally, print BLEU score for each example 
for i, bleu in enumerate(bleu_scores): 
    print(f'Example {i + 1} BLEU Score: {bleu}') 
 
# Save the tokenizer for later use during inference 
tokenizer.save('G:/TextSummModel/tokenizer/tokenizer.pkl') 
 
# Save the trained model for later use in generating summaries during inference 
seq2seq_Model.save('G:/TextSummModel/models/modelsseq2seq_model_final.h5') 
 
# Load the saved tokenizer and model during inference 
loaded_tokenizer = Tokenizer() 
loaded_tokenizer = loaded_tokenizer.load('G:/TextSummModel/tokenizer/tokenizer.pkl') 
 
loaded_model = tf.keras.models.load_model('G:/TextSummModel/models/modelsseq2seq_model_final.h5') 
 
# Example of using the loaded model for inference 
sample_input_text = "Input text for summarization." 
sample_input_sequence = loaded_tokenizer.texts_to_sequences([sample_input_text]) 
sample_padded_sequence = pad_sequences(sample_input_sequence, maxlen=max_len, padding='post') 
 
# Dummy initial input for the decoder during inference 
dummy_decoder_input = np.zeros((1, max_len)) 
 
# Generate a summary using the loaded model 
predicted_summary = loaded_model.predict([sample_padded_sequence, dummy_decoder_input]) 
 
# Convert the predicted summary back to text 
predicted_summary_text = tokenizer.sequences_to_texts([np.argmax(token) for token in predicted_summary[0]]) 
 
print(f'Sample Input Text: {sample_input_text}') 
print(f'Predicted Summary: {predicted_summary_text}')