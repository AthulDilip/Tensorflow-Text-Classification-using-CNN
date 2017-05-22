# Tensorflow-Text-Classification-using-CNN
This is an implementation of a Convolutional Neural Network for Text Classification in Tensorflow. The model used above is a special type of CNN in which convolution is performed on the input three times with different filters and then combined together and is followed by a fully connected output layer. 

The model performed well when tested on IMDB's movie reviews dataset. The model was trained to classify weather a review is positive or negative

## How to train the model
To train the model run the following command

```sh
python classifier.py --image_path <path-to-image-file>
```

## How to use the model after training
When training a model will be saved inside the runs folder at each 100th step. To use the trained model call the function classify inside textclassifier file

```python
import text_classifier
result = text_classifier.classify(checkpoint_dir, x_text)
```

where checkpoint_dir is the directory in which the model are saved and x_text is the text to be classified. (refer usage_example.py for an example of the usage)

