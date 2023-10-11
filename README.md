# emotion-recognition

This is an epic primary emotion prediction model that I've built for you. It predicts the primary emotion of a given tweet. Let me break it down for you:

## Usage

To use this model, you need to follow these steps:

1. Import the necessary libraries and functions:

2. 2. Define the `pe_predict` function with the following parameters:
- `test_file`: The test data file containing the tweets to predict emotions for.
- `model`: The trained emotion prediction model.
- `tokenizer`: The tokenizer to encode the text.
- `max_length`: The maximum length of the input text (default is 128).
- `threshold`: The threshold value for prediction (default is 0.5).

3. Inside the `pe_predict` function, perform the following steps:
- Read the test data file and extract the tweets and local IDs.
- Set the device to CPU and move the model to the CPU device.
- Tokenize the tweets using the tokenizer and store them in a list.
- Iterate over the tokenized tweets and create input tensors for each tweet.
- Make predictions for each input tensor using the trained model.
- Convert the predicted probabilities to labels using a label dictionary.
- Store the predictions and probabilities in separate lists.
- Create a CSV file with the local IDs, tweets, and primary emotions.
- Create a DataFrame from the CSV data.

4. Finally, save the predictions to a CSV file and return the DataFrame.

## Example

Here's an example of how to use the `pe_predict` function:

```python
# Load the trained model and tokenizer
model = YourModel()
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

# Load the test data
test_data = pd.read_csv('test_data.csv')

# Call the pe_predict function
predictions = pe_predict(test_data, model, tokenizer)

# Print the predictions DataFrame
print(predictions)

