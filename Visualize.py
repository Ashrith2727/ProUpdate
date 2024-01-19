# Code for Visualizing Attention Weights during Inference 
 
# Extract attention weights from the model 
attention_weights_model = Model(inputs=seq2seq_Model.inputs, 
                                outputs=[seq2seq_Model.get_layer('attention').output]) 
 
# Function to visualize attention weights 
def visualize_attention(input_text, target_summary): 
    input_sequence = loaded_tokenizer.texts_to_sequences([input_text]) 
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post') 
 
    target_sequence = loaded_tokenizer.texts_to_sequences([target_summary]) 
    padded_target_sequence = pad_sequences(target_sequence, maxlen=max_len, padding='post') 
 
    attention_weights = attention_weights_model.predict([padded_input_sequence, padded_target_sequence]) 
 
    # Visualization code (e.g., heatmap plot) can be added here 
 
# Example of using the visualization function 
sample_input_text = "Input text for summarization." 
sample_target_summary = "Target summary for the input text." 
 
visualize_attention(sample_input_text, sample_target_summary) 
 
# Save Attention Weights for Later Analysis 
attention_weights = attention_weights_model.predict([sample_padded_sequence, dummy_decoder_input]) 
np.save('G:/TextSummModel/attention_weights/attention_weights.npy', attention_weights) 
 
# Load Attention Weights for Analysis 
loaded_attention_weights = np.load('G:/TextSummModel/attention_weights/attention_weights.npy') 