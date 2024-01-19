# Code for Extracting and Visualizing Word Embeddings 
 
# Extract word embeddings from the embedding layer of the trained model 
embedding_weights = seq2seq_Model.get_layer('embedding').get_weights()[0] 
 
# Function to visualize word embeddings using t-SNE 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
 
def visualize_word_embeddings(word_index, embedding_weights, num_words_to_visualize=100): 
    words_to_visualize = list(word_index.keys())[:num_words_to_visualize] 
    indices_to_visualize = [word_index[word] for word in words_to_visualize] 
 
    # Extract embeddings for selected words 
    selected_embeddings = embedding_weights[indices_to_visualize] 
 
    # Apply t-SNE for dimensionality reduction 
    tsne = TSNE(n_components=2, random_state=42) 
    embedded_words = tsne.fit_transform(selected_embeddings) 
 
    # Visualization code (scatter plot) can be added here 
    plt.figure(figsize=(10, 8)) 
    plt.scatter(embedded_words[:, 0], embedded_words[:, 1], c='blue', alpha=0.5) 
    for i, word in enumerate(words_to_visualize): 
        plt.annotate(word, (embedded_words[i, 0], embedded_words[i, 1]), alpha=0.8) 
 
    plt.title('t-SNE Visualization of Word Embeddings') 
    plt.xlabel('t-SNE Dimension 1') 
    plt.ylabel('t-SNE Dimension 2') 
    plt.show() 
 
# Example of using the visualization function 
visualize_word_embeddings(word_index, embedding_weights) 
 
# Code for Extracting and Saving Encoder Representations 
 
# Extract encoder representations (outputs of the last LSTM layer in the encoder) 
encoder_representation_model = Model(inputs=seq2seq_Model.inputs, 
                                      outputs=[seq2seq_Model.get_layer('bidirectional_lstm_2').output]) 
 
# Function to extract encoder representations for a given input text 
def extract_encoder_representations(input_text): 
    input_sequence = loaded_tokenizer.texts_to_sequences([input_text]) 
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post') 
 
    # Dummy initial input for the decoder during inference 
    dummy_decoder_input = np.zeros((1, max_len)) 
 
    # Extract encoder representations 
    encoder_representations = encoder_representation_model.predict([padded_input_sequence, 
dummy_decoder_input]) 
 
    return encoder_representations 
 
# Example of using the function 
sample_input_text = "Input text for summarization." 
encoder_representations = extract_encoder_representations(sample_input_text) 
print(f'Encoder Representations Shape: {encoder_representations.shape}') 