# Sentence Classification: Character Based Deep Learning
## Data Understanding, Idea and Data Preparation
The text is unclear and not human understandable. If it would have words from any language (human understandable), then I would have used the Word2Vec model to generate vectors for this text. However, I am  using character level information (vectors) to train the model for classification.

The idea came from the paper titled \href{https://arxiv.org/pdf/1502.01710.pdf}{Text Understanding from Scratch}. The author demonstrates text understanding from character level inputs, using convolutional neural network.

In the next step, I try to understand train and test text. I find 83 and 80 unique characters in the train and test text respectively. I also see the frequency of each character in the both texts. The extra characters (3 in train text) in the train text are infrequent and, there are other 2 characters which are very infrequent in the both texts. Due to infrequent occurrence, these characters do not help much in learning the model. Therefore, I replace all these characters with one unique character. Now, we remain with 78 unique characters in the text. I also find average and maximum character length of each sentence. The 453 and 415 are the maximum and the average character length of sentences from the text respectively. I also find the distribution of 12 classes from the train data as. The distribution of classes is not uniform as shown in ![class_distribution](https://user-images.githubusercontent.com/17526799/28752285-cea09dc4-751b-11e7-830f-ccd633f490c2.png).

In the next step, I prepare data for training. First, I create a dictionary of characters with unique ids (numbers from 1 to 78, since we have 78 unique characters). Then, I represent each character of the sentence with it\textquotesingle s id (same character with same id). Then, I create a vector representation of each sentence of length 453 (maximum character length sentence). The sentences which do not have 453 characters, I do padding with zeros to make it of length 453. Finally, we have training data (vector representation) of dimensions (32513,453). Then, I divide training data into train (90%)and dev (10%) data. The dev data is used to test the accuracy of the model during training.

## Approach
I use train data (90%) for training the models and dev data (10\%) for testing the accuracy on it.
### Convolutional Neural Network
In this architecture, I use different hyper-parameters like learning rate, epoch, batch size, dropout. I also use kernels of different dimensions.  
#### conv2d-$>$maxpool2d-$>$fully connected
First, I start with one convolution-maxpooling layer pair followed by one fully connected layer. I run the model with different hyper-parameters. I get an accuracy of around 19% after 10 epochs for hyper-parameters, learning rate=0.01 and batch size=200.
#### conv2d-$>$maxpool2d-$>$conv2d-$>$maxpool2d-$>$fully connected
Then, I extend the model with the addition of one more convolution-maxpooling layer pair. I run the model with different hyper-parameters. I still get accuracy around 23% for the hyper-parameters, learning rate=0.01 and batch size=300.
### Recurrent Neural Network
In this architecture, I use different hyper-parameters like learning rate, epoch, batch size, dropout, rnn cells (rnn, lstm, gru), number of layers and number of hidden units. In this approach, the model learns embedding for each character at each step. Finally, this embedding weight is used for predictions.
#### LSTM and GRU layers with generation of embedding
I start with single LSTM and GRU layers. I run the model with different hyper-parameters. The model accuracy still does not increase from previous models. Then I extend the model 2 LSTM and GRU layers, the accuracy remains the constant (21\%) for different for epochs.
#### LSTM and GRU layers with one hot vector representation of each character
Till now, I used a unique id for each character in vector representation. Now, I use one hot vector for each character of length 78 (since we have 78 unique characters). I run the RNN models of previous configurations with new this vector representation. The accuracy gets increase to 27%. The best hyper-parameters are learning rate=0.01, epochs=25, hidden units=300 and batch size=200.
### Classical Classification Approaches
Since the accuracy of the models is below par. Therefore, I thought to try with some classical classification approaches. I train the model with logistic regression, still I get the accuracy around 15% for different epochs. I also train the model with random forest classifiers for different number of estimators (tress), I get the best accuracy of 18% on dev data.
## Conclusion
I tried different models based on character level vector representation. I also used 2 types of vector representations (unique id for each character and one hot vector of length 78 for each character). I got the best accuracy of 27% from one layer LSTM model using one hot vector representation. The accuracy is par below expected accuracy. I also see the model predicts the majority class labels (mostly classes 6,7 and 3). But, this might be due to the nature of the data itself or the data was not enough to learn the features.

If it was some human readable representation of text, then I would have gone through the text and might have some actual insight of the data. 
