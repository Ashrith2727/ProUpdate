# Code for Extracting and Saving Attention Weights during Inference 
 
# Function to extract and save attention weights for a given input text and target summary 
def extract_and_save_attention_weights(input_text, target_summary): 
    input_sequence = loaded_tokenizer.texts_to_sequences([input_text]) 
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post') 
 
    target_sequence = loaded_tokenizer.texts_to_sequences([target_summary]) 
    padded_target_sequence = pad_sequences(target_sequence, maxlen=max_len, padding='post') 
 
    # Dummy initial input for the decoder during inference 
    dummy_decoder_input = np.zeros((1, max_len)) 
37 
 
 
    # Extract attention weights 
    attention_weights = attention_weights_model.predict([padded_input_sequence, padded_target_sequence]) 
 
    # Save attention weights for later analysis 
    np.save('G:/TextSummModel/attention_weights/attention_weights_custom.npy', attention_weights) 
  
# Example of using the function 
sample_input_text = "Input text for summarization." 
sample_target_summary = "Target summary for the input text." 
 
extract_and_save_attention_weights(sample_input_text, sample_target_summary) 
 
# Code for Generating Summaries for Multiple Input Texts in a Batch 
 
# Function to generate summaries for a batch of input texts 
def generate_summaries_batch(input_texts): 
    input_sequences = loaded_tokenizer.texts_to_sequences(input_texts) 
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post') 
 
    # Dummy initial input for the decoder during inference 
    dummy_decoder_input = np.zeros((len(input_texts), max_len)) 
 
    # Generate summaries for the batch 
    batch_summaries = loaded_model.predict([padded_input_sequences, dummy_decoder_input]) 
 
    # Convert the predicted summaries back to text 
    batch_summaries_text = [] 
    for summary in batch_summaries: 
        batch_summaries_text.append(tokenizer.sequences_to_texts([np.argmax(token) for token in summary])) 
 
    return batch_summaries_text 
 
# Example of using the function 
batch_input_texts = ["Input text 1 for summarization.", "Input text 2 for summarization.", "Input text 3 for 
summarization."] 
batch_predicted_summaries = generate_summaries_batch(batch_input_texts) 
 
# Print the generated summaries for each input text in the batch 
for i, summary in enumerate(batch_predicted_summaries): 
    print(f'Input Text {i + 1} - Generated Summary: {summary}')