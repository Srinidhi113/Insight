"""Bounded, deterministic ingestion without external tokenizer downloads."""
import html
import json
import re
from .schemas import AnalyzeRequest

MAX_CHARS = 250_000
MAX_SENTENCES = 1500


def discussion_blocks(value):
    """Accept Reddit comment trees, API Listing wrappers, or arrays of comments."""
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, list):
        for child in value:
            yield from discussion_blocks(child)
    elif isinstance(value, dict):
        for key in ('title', 'selftext', 'body', 'text'):
            if isinstance(value.get(key), str) and value[key].strip():
                yield value[key]
        for key in ('comments', 'replies', 'children', 'data'):
            if key in value and isinstance(value[key], (dict, list)):
                yield from discussion_blocks(value[key])


def preprocess(request: AnalyzeRequest):
    if request.text is not None:
        raw = request.text
    else:
        if len(json.dumps(request.discussion)) > MAX_CHARS:
            raise ValueError('JSON input exceeds 250,000 characters.')
        raw = '\n'.join(discussion_blocks(request.discussion))
    if len(raw) > MAX_CHARS:
        raise ValueError('Input exceeds 250,000 characters.')
    raw = html.unescape(raw)
    raw = re.sub(r'<[^>]*>', ' ', raw)
    raw = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', raw)
    raw = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', raw)
    raw = re.sub(r'https?://\S+|www\.\S+', ' ', raw)
    raw = re.sub(r'(?<!\w)/?u/[\w-]+', ' ', raw)
    raw = re.sub(r'\b\d+\s+(?:points?|votes?)\b|\b\d+\s+(?:hours?|days?|minutes?)\s+ago\b', ' ', raw, flags=re.I)
    sentences, seen, duplicate_count, short_count = [], set(), 0, 0
    for line in raw.splitlines():
        line = re.sub(r'^\s*(?:>+|#{1,6}|[-*+]\s|\d+\.\s)\s*', '', line)
        line = re.sub(r'^\s*\[?\d{4}-\d{2}-\d{2}[T ][\d:.+Z-]+\]?\s*', '', line)
        line = re.sub(r'[*_`~]', '', line)
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\[])', line.strip()):
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            if sentence.lower() in ('[deleted]', '[removed]') or len(re.findall(r'\b\w+\b', sentence)) < 4:
                short_count += bool(sentence)
                continue
            # Long comments are segmented rather than silently truncated by MPNet.
            words = sentence.split()
            for start in range(0, len(words), 100):
                fragment = ' '.join(words[start:start+100])
                key = re.sub(r'\W+', ' ', fragment.lower()).strip()
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                sentences.append({'id': len(sentences), 'text': fragment})
                if len(sentences) > MAX_SENTENCES:
                    raise ValueError('Too many sentences (limit 1,500). Split the discussion into smaller inputs.')
    if not sentences:
        raise ValueError('No usable English discussion text. Include sentences with at least four words.')
    return {'sentences': sentences, 'input_characters': len(raw), 'exact_duplicates_removed': duplicate_count, 'short_fragments_removed': short_count}
