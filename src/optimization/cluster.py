import pandas as pd
from dbscan import DBSCAN
from sentence_transformers import SentenceTransformer

from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


device = "cuda:0"


def test_features():
    # Example texts
    original_text = "This is the original text. It contains several sentences with various complexities."
    rewritten_text = "This rewritten version simplifies the original content, making it more accessible."

    # Sentence count
    original_sentence_count = len(TextBlob(original_text).sentences)
    rewritten_sentence_count = len(TextBlob(rewritten_text).sentences)

    # Average word length
    original_avg_word_length = sum(len(word) for word in word_tokenize(original_text)) / len(word_tokenize(original_text))
    rewritten_avg_word_length = sum(len(word) for word in word_tokenize(rewritten_text)) / len(word_tokenize(rewritten_text))

    # Sentiment analysis
    original_sentiment = TextBlob(original_text).sentiment.polarity
    rewritten_sentiment = TextBlob(rewritten_text).sentiment.polarity

    # TF-IDF Vectorization example
    vectorizer = TfidfVectorizer()
    corpus = [original_text, rewritten_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Extracting Features
    features = {
        "original_sentence_count": original_sentence_count,
        "rewritten_sentence_count": rewritten_sentence_count,
        "original_avg_word_length": original_avg_word_length,
        "rewritten_avg_word_length": rewritten_avg_word_length,
        "original_sentiment": original_sentiment,
        "rewritten_sentiment": rewritten_sentiment,
        # Add more features as needed
    }



def run():
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)
    df = pd.read_csv("./fitted_conditioned_prompts.csv")
    
    org_text = df["original_text"].tolist()
    rew_text = df["rewrite_prompt"].tolist()
    
    text_list = [f"{org}\n{rew}" for org, rew in zip(org_text, rew_text)]

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

    pca = PCA(n_components=16)
    embds_pca = pca.fit_transform(embds)
    
    labels, core_samples_mask = DBSCAN(embds_pca, eps=0.5, min_samples=5)
    
    print(labels)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    run()

