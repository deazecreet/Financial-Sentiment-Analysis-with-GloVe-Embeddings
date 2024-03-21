# Financial-Sentiment-Analysis-with-GloVe-Embeddings

## Project Overview
This project delves into Natural Language Processing (NLP) to perform sentiment analysis on financial texts. Using the `financial_phrasebank` dataset from HuggingFace, we classify sentences into negative, neutral, and positive sentiments. The model leverages GloVe embeddings to enhance feature representation of text data, significantly improving classification accuracy.

## Dataset
The dataset comprises sentences from financial news, labeled with sentiments by financial experts. Only sentences with a unanimous agreement among annotators are included, ensuring high label quality. This project aims to automate sentiment classification, aiding in financial decision-making processes.

## Technologies Used
- Python for data processing and modeling.
- TensorFlow and Keras for building the LSTM neural network.
- GloVe for pre-trained word embeddings.
- Pandas and NumPy for data manipulation.
- Matplotlib and Seaborn for data visualization.

## Model Architecture
The model is a Sequential LSTM network, utilizing GloVe pre-trained embeddings to transform text into meaningful representations. The LSTM layer captures sequential dependencies, followed by a Dense layer for classification.

## Challenges and Solutions
- **Text Normalization:** Implemented functions to lowercase texts, remove numbers, punctuations, and currency-related terms, and expand slangs, standardizing the input data.
- **Class Imbalance:** Utilized categorical cross-entropy loss to handle the imbalance effectively.
- **Overfitting:** Introduced Dropout layers and applied an Early Stopping callback based on validation accuracy.

## Results
The model achieved an impressive validation accuracy, demonstrating the potential of combining LSTM with GloVe embeddings for sentiment analysis in financial texts. Detailed performance metrics are provided, including a classification report showcasing precision, recall, and F1-scores for each class.

## Future Work
- Experimenting with different neural network architectures and hyperparameters.
- Expanding the dataset to include more diverse financial texts.
- Exploring alternative pre-trained embeddings like BERT and Word2Vec.

## Acknowledgments
- HuggingFace for providing the `financial_phrasebank` dataset.
- Stanford University for the GloVe embeddings.
- The TensorFlow and Keras teams for their excellent deep learning libraries.
