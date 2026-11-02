import time
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import get_models
from .preprocessing import preprocess

@contextmanager
def timed(timings, name):
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 4)


def detect_topics(sentences, vectors, requested):
    k = min(requested, len(sentences), len(np.unique(np.round(vectors, 5), axis=0)))
    if k == 1:
        labels = np.zeros(len(sentences), dtype=int)
    else:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors)
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    # Stable display order by first source appearance, rather than arbitrary KMeans IDs.
    members = sorted(groups.values(), key=lambda x: x[0])
    documents = [' '.join(sentences[i]['text'] for i in ids) for ids in members]
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b')
        weights = vectorizer.fit_transform(documents).toarray()
        terms = vectorizer.get_feature_names_out()
        keywords = [[str(terms[j]) for j in np.argsort(-row) if row[j] > 0][:5] for row in weights]
    except ValueError:
        keywords = [[] for _ in documents]
    return [{'id': n+1, 'label': ' / '.join(keys[:3]).title() or f'Topic {n+1}', 'keywords': keys, 'indices': ids} for n, (ids, keys) in enumerate(zip(members, keywords))]


def select_sources(indices, vectors, threshold, refine):
    if not refine or len(indices) < 2:
        return indices[:]
    local = vectors[indices]
    centroid = local.mean(axis=0, keepdims=True)
    ranks = np.argsort(-cosine_similarity(local, centroid).ravel())
    selected = []
    for pos in ranks:
        idx = indices[int(pos)]
        if not selected or cosine_similarity(vectors[idx:idx+1], vectors[selected]).max() < threshold:
            selected.append(idx)
    return sorted(selected)


def make_chunks(texts, models, budget):
    """Adjacent TF-IDF similarity boundaries; preserve order and enforce model tokens."""
    limit = min(budget, getattr(models, 'input_limit', 900))
    fragments = []
    for text in texts:
        tokens = models.tokens(text)
        # Subdivide oversized fragments using the actual model tokenizer.
        for start in range(0, len(tokens), limit-8):
            fragment = models.decode(tokens[start:start+limit-8])
            # Re-tokenization may expand at byte boundaries: halve recursively if needed.
            if len(models.tokens(fragment)) > limit:
                mid = max(1, len(tokens[start:start+limit-8]) // 2)
                part = tokens[start:start+limit-8]
                fragments.extend([models.decode(part[:mid]), models.decode(part[mid:])])
            else:
                fragments.append(fragment)
    try:
        matrix = TfidfVectorizer(stop_words='english').fit_transform(fragments)
        similarity = [1.] + [float(cosine_similarity(matrix[i-1], matrix[i])[0, 0]) for i in range(1, len(fragments))]
    except ValueError:
        similarity = [1.] * len(fragments)
    chunks, current = [], ''
    for i, fragment in enumerate(fragments):
        candidate = f'{current} {fragment}'.strip()
        boundary = current and similarity[i] < .08 and len(models.tokens(current)) >= limit // 2
        if current and (len(models.tokens(candidate)) > limit or boundary):
            chunks.append(current); current = fragment
        else:
            current = candidate
    if current:
        chunks.append(current)
    if any(len(models.tokens(chunk)) > limit for chunk in chunks):
        raise ValueError('Unable to split input safely within the model token limit.')
    return chunks


def summarize_texts(texts, models, options):
    chunks = make_chunks(texts, models, options.chunk_tokens)
    first_pass = [models.summarize(chunk, min(options.summary_tokens, max(24, len(models.tokens(chunk)) // 2))) for chunk in chunks]
    intermediate = [{'input': chunk, 'summary': summary} for chunk, summary in zip(chunks, first_pass)]
    if not options.second_pass:
        return {'summary': ' '.join(first_pass), 'chunks': intermediate, 'compression_rounds': 0}
    # Hierarchical integration avoids truncating intermediate summaries on long inputs.
    merged = ' '.join(first_pass)
    limit = min(options.chunk_tokens, getattr(models, 'input_limit', 900))
    rounds = 0
    while len(models.tokens(merged)) > limit:
        old_length = len(models.tokens(merged))
        parts = make_chunks([merged], models, limit)
        merged = ' '.join(models.summarize(part, max(16, min(options.summary_tokens, len(models.tokens(part)) // 2))) for part in parts)
        rounds += 1
        if len(models.tokens(merged)) >= old_length or rounds > 16:
            raise RuntimeError('Summarization did not converge within token budget. Reduce input size.')
    final = models.summarize(merged, options.summary_tokens)
    return {'summary': final, 'chunks': intermediate, 'compression_rounds': rounds + 1}


def analyze(request, progress=lambda stage: None):
    start = time.perf_counter(); timings = {}
    progress('Cleaning and segmenting discussion')
    with timed(timings, 'preprocessing'):
        prepared = preprocess(request)
    progress('Loading models' if request.options.mode == 'ai' else 'Preparing offline demo')
    with timed(timings, 'model_loading'):
        models = get_models(request.options.mode)
    progress('Generating sentence representations')
    with timed(timings, 'embedding'):
        sentences = prepared['sentences']
        vectors = models.embed([s['text'] for s in sentences])
    progress('Clustering discussion topics')
    with timed(timings, 'topic_detection'):
        topics = detect_topics(sentences, vectors, request.options.num_topics)
    with timed(timings, 'summarization'):
        for topic in topics:
            progress(f'Summarizing topic {topic["id"]} of {len(topics)}')
            ids = topic.pop('indices')
            selected = select_sources(ids, vectors, request.options.dedup_threshold, request.options.semantic_refinement)
            result = summarize_texts([sentences[i]['text'] for i in selected], models, request.options)
            topic.update(result)
            topic['sources'] = [dict(sentences[i], selected_for_summary=i in selected) for i in ids]
            topic['sentence_count'] = len(ids)
            topic['selected_sentence_count'] = len(selected)
            topic['summary_word_count'] = len(result['summary'].split())
    warnings = []
    if request.options.mode == 'demo':
        warnings.append('Offline demo uses lexical clustering and extractive summaries. Choose AI mode for MPNet + DistilBART.')
    if len(topics) < request.options.num_topics:
        warnings.append('Topic count reduced to the number of distinct sentence representations.')
    warnings.append('Generated summaries can omit context or contain errors. Inspect source evidence before relying on them.')
    return {'mode': request.options.mode, 'models': models.name, 'options': request.options.model_dump(), 'topics': topics,
            'preprocessing': prepared, 'timings_seconds': timings, 'elapsed_seconds': round(time.perf_counter()-start, 4), 'warnings': warnings}


def plain_text(result):
    lines = ['InsightFlow', f'Mode: {result["mode"]}', f'Models: {result["models"]}', '']
    lines.extend(result['warnings']); lines.append('')
    for topic in result['topics']:
        lines.extend([f'Topic {topic["id"]}: {topic["label"]}', 'Keywords: '+', '.join(topic['keywords']), topic['summary'], ''])
    return '\n'.join(lines)
