def beam_search(input_text, beam_width=3): 
    input_sequence = loaded_tokenizer.texts_to_sequences([input_text]) 
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post') 
 
    # Dummy initial input for the decoder during inference 
    dummy_decoder_input = np.zeros((1, max_len)) 
 
    # Generate initial predictions 
    initial_predictions = loaded_model.predict([padded_input_sequence, dummy_decoder_input]) 
 
    # Start beam search 
    sequences = [[list(), 0.0]] 
    for i in range(beam_width): 
        seq, score = sequences[i] 
35 
 
        for j in range(len(initial_predictions[0])): 
            candidate = [seq + [j], score - np.log(initial_predictions[0][j])] 
            sequences.append(candidate) 
 
    # Sort sequences by score 
    sequences.sort(key=lambda x: x[1]) 
 
    # Return the sequence with the highest score 
    best_sequence = sequences[0][0] 
    predicted_summary = tokenizer.sequences_to_texts(best_sequence) 
    return predicted_summary 
 
# Example of using beam search 
sample_input_text = "Input text for summarization." 
beam_search_summary = beam_search(sample_input_text) 
print(f'Beam Search Summary: {beam_search_summary}') 
 
# Code for Evaluating the Model on Custom Input 
 
def evaluate_custom_input(input_text, target_summary): 
    input_sequence = loaded_tokenizer.texts_to_sequences([input_text]) 
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post') 
 
    target_sequence = loaded_tokenizer.texts_to_sequences([target_summary]) 
    padded_target_sequence = pad_sequences(target_sequence, maxlen=max_len, padding='post') 
 
    # Evaluate the model on custom input 
    custom_input_loss, custom_input_accuracy = seq2seq_Model.evaluate([padded_input_sequence, 
padded_target_sequence], 
                                                                      padded_target_sequence) 
    print(f'Custom Input Loss: {custom_input_loss}, Custom Input Accuracy: {custom_input_accuracy}') 
 
# Example of using the evaluation function 
custom_input_text = "Custom input text for evaluation." 
custom_target_summary = "Target summary for the custom input text." 
 
evaluate_custom_input(custom_input_text, custom_target_summary) 