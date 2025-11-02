def preprocess_titles(articles):
    """Minimal preprocessing before tokenization."""
    return [article.title.strip() for article in articles]
