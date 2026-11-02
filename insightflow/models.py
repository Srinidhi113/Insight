"""Lazy, process-cached models. AI errors never silently switch to demo mode."""
import os
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDING_MODEL = os.getenv('INSIGHTFLOW_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
SUMMARY_MODEL = os.getenv('INSIGHTFLOW_SUMMARY_MODEL', 'sshleifer/distilbart-cnn-12-6')

class DemoModels:
    name = 'demo: TF-IDF + extractive sentence selection (no neural models)'
    def embed(self, texts):
        try:
            return TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=12000).fit_transform(texts).toarray()
        except ValueError:
            return TfidfVectorizer(analyzer='char', ngram_range=(2, 3)).fit_transform(texts).toarray()

    def tokens(self, text):
        return text.split()

    def decode(self, tokens):
        return ' '.join(tokens)

    def summarize(self, text, max_tokens):
        import re
        units = re.split(r'(?<=[.!?])\s+', text)
        vectors = self.embed(units)
        scores = vectors @ vectors.mean(axis=0)
        selected, size = [], 0
        for i in np.argsort(-scores):
            count = len(self.tokens(units[i]))
            if size + count <= max_tokens:
                selected.append(int(i)); size += count
        if not selected:
            return self.decode(self.tokens(text)[:max_tokens])
        return ' '.join(units[i] for i in sorted(selected))

class AIModels:
    name = f'{EMBEDDING_MODEL} + {SUMMARY_MODEL}'
    def __init__(self):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise RuntimeError('AI packages are missing. Install requirements-ai.txt, then restart the server.') from exc
        self.torch = torch
        self.device = os.getenv('INSIGHTFLOW_DEVICE', 'cpu')
        if self.device == 'cpu':
            torch.set_num_threads(max(1, int(os.getenv('INSIGHTFLOW_CPU_THREADS', str(min(4, os.cpu_count() or 1))))))
        try:
            self.encoder = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError('Could not load AI models. Check internet access, available memory and the Hugging Face cache. Run python scripts/download_models.py; see README.') from exc
        self.input_limit = min(int(getattr(self.model.config, 'max_position_embeddings', 1024)), 1024) - 4

    def embed(self, texts):
        return self.encoder.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def summarize(self, text, max_tokens):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=False).to(self.device)
        if inputs['input_ids'].shape[1] > self.input_limit + 2:
            raise ValueError('Internal token budget exceeded; input was not truncated.')
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                min_new_tokens=min(24, max(1, len(self.tokens(text)) // 4)),
                num_beams=4, do_sample=False, no_repeat_ngram_size=3, early_stopping=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

@lru_cache(maxsize=2)
def get_models(mode):
    return DemoModels() if mode == 'demo' else AIModels()
