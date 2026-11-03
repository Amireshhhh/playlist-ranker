"""Lyrics analytics: word counts, distinctive words, vocabulary, repetition, sentiment, themes.

All measures are computed locally (no paid NLP service). Sentiment uses VADER
(Hutto & Gilbert, 2014), a lexicon model that runs offline. Lyrics text never leaves
the machine and is never written into the dashboard; only counts and scores are.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

SECTION = re.compile(r"^\s*[\[\(][^\]\)]*[\]\)]\s*$")  # [Chorus], (Verse 2)
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd
we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with
won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
just like get got go gonna wanna gotta ain't cause 'cause cuz let yeah yeh oh ooh ah aah uh huh hey ha la na
da woah whoa mm mmm hmm eh ay ayy yo ya im ill ive dont cant wont youre thats its id isnt doesnt didnt
one now will still back know say said see make made take come came way thing things every ever even also
""".split())

PRONOUN_GROUPS = {
    "I / me": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd", "im", "ive"},
    "you": {"you", "your", "yours", "yourself", "you're", "you've", "you'll", "you'd", "ya", "youre"},
    "we / us": {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'll"},
    "they / them": {"they", "them", "their", "theirs", "themselves", "they're", "they've", "he", "she",
                    "him", "her", "his"},
}


def clean_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not SECTION.match(ln)]


def tokenize(text: str) -> List[str]:
    return [w.replace("’", "'") for w in WORD.findall(text.lower())]


def content_words(tokens: List[str], extra_stop: frozenset = frozenset()) -> List[str]:
    return [w for w in tokens if len(w) > 1 and w not in STOPWORDS and w not in extra_stop]


def mattr(tokens: List[str], window: int = 50) -> Optional[float]:
    """Moving-average type-token ratio (Covington & McFall, 2010). Unlike the plain TTR it
    does not fall just because a song is longer. None when the song is shorter than the window."""
    n = len(tokens)
    if n < window:
        return None
    counts = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, n):
        out, inc = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] += 1
        total += len(counts)
    return total / (n - window + 1) / window


def repetitiveness(text: str) -> Optional[float]:
    """1 - compressed_size / raw_size with zlib (LZ77 + Huffman). Repeated choruses compress
    well, so higher means more repetitive (the measure of C. Morris, 'Are Pop Lyrics Getting
    More Repetitive?', 2017). None for very short texts, where header overhead dominates."""
    raw = "\n".join(clean_lines(text)).lower().encode("utf-8")
    if len(raw) < 200:
        return None
    return 1.0 - len(zlib.compress(raw, 9)) / len(raw)


_vader = None


def sentiment(text: str) -> Dict[str, float]:
    """Mean VADER compound score over lines (in [-1, 1]) and the share of positive / negative lines."""
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    lines = clean_lines(text)
    if not lines:
        return {"compound": 0.0, "pos_lines": 0.0, "neg_lines": 0.0}
    scores = [_vader.polarity_scores(ln)["compound"] for ln in lines]
    return {
        "compound": float(np.mean(scores)),
        "pos_lines": sum(s >= 0.05 for s in scores) / len(scores),
        "neg_lines": sum(s <= -0.05 for s in scores) / len(scores),
    }


@dataclass
class SongLyrics:
    index: int
    title: str
    artist: str
    n_words: int
    n_unique: int
    mattr: Optional[float]
    repetitiveness: Optional[float]
    sentiment: Dict[str, float]
    top_words: List[tuple]
    distinctive: List[tuple] = field(default_factory=list)
    pronouns: Dict[str, int] = field(default_factory=dict)


@dataclass
class LyricsReport:
    songs: List[SongLyrics]
    word_counts: Counter
    bigrams: Counter
    pronouns: Dict[str, int]
    by_artist: Dict[str, Dict[str, float]]
    tfidf_matrix: Optional[np.ndarray]      # songs x terms, L2-normalised rows
    tfidf_terms: List[str]
    themes: List[Dict]                       # NMF topics
    total_words: int

    @property
    def vocabulary(self) -> int:
        return len(self.word_counts)


def analyze(tracks, n_themes: int = 4) -> Optional[LyricsReport]:
    """Analyse the tracks that have ``.lyrics``. Returns None when none do."""
    with_lyrics = [(i, t) for i, t in enumerate(tracks) if t.lyrics and not t.instrumental]
    if not with_lyrics:
        return None
    # Artist names show up in lyric transcriptions (shout-outs, 'Artist:' credits) and would
    # top the counts; drop them. Title words stay: a chorus that repeats the title is real signal.
    name_words = frozenset(w for _, t in with_lyrics for w in tokenize(t.artist) if len(w) > 2)
    songs, docs = [], []
    wc, bg, pron = Counter(), Counter(), Counter()
    for i, t in with_lyrics:
        toks = tokenize("\n".join(clean_lines(t.lyrics)))
        cw = content_words(toks, name_words)
        c = Counter(cw)
        wc.update(c)
        for ln in clean_lines(t.lyrics):
            lw = content_words(tokenize(ln), name_words)
            bg.update(f"{a} {b}" for a, b in zip(lw, lw[1:]))
        p = {g: sum(1 for w in toks if w in ws) for g, ws in PRONOUN_GROUPS.items()}
        pron.update(p)
        songs.append(SongLyrics(
            index=i, title=t.title, artist=t.artist, n_words=len(toks), n_unique=len(set(toks)),
            mattr=mattr(toks), repetitiveness=repetitiveness(t.lyrics), sentiment=sentiment(t.lyrics),
            top_words=c.most_common(8), pronouns=p,
        ))
        docs.append(" ".join(cw))

    matrix, terms, themes = None, [], []
    if len(docs) >= 2 and any(docs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(token_pattern=r"\S+", min_df=1, max_df=0.9 if len(docs) >= 5 else 1.0,
                              sublinear_tf=True)
        try:
            X = vec.fit_transform(docs)
        except ValueError:  # every term pruned
            X = None
        if X is not None and X.shape[1]:
            terms = list(vec.get_feature_names_out())
            matrix = X.toarray()
            for s, row in zip(songs, matrix):
                top = np.argsort(-row)[:6]
                s.distinctive = [(terms[j], round(float(row[j]), 3)) for j in top if row[j] > 0]
            themes = _themes(X, terms, songs, n_themes)

    by_artist: Dict[str, Dict[str, float]] = {}
    for a in dict.fromkeys(s.artist for s in songs):
        ss = [s for s in songs if s.artist == a]
        m = [s.mattr for s in ss if s.mattr is not None]
        r = [s.repetitiveness for s in ss if s.repetitiveness is not None]
        by_artist[a] = {
            "songs": len(ss),
            "words_per_song": float(np.mean([s.n_words for s in ss])),
            "mattr": float(np.mean(m)) if m else float("nan"),
            "repetitiveness": float(np.mean(r)) if r else float("nan"),
            "sentiment": float(np.mean([s.sentiment["compound"] for s in ss])),
        }
    return LyricsReport(songs=songs, word_counts=wc, bigrams=bg, pronouns=dict(pron), by_artist=by_artist,
                        tfidf_matrix=matrix, tfidf_terms=terms, themes=themes,
                        total_words=sum(s.n_words for s in songs))


def _themes(X, terms, songs, n_themes) -> List[Dict]:
    """Non-negative matrix factorisation of the TF-IDF matrix: each component is a 'theme'
    (a weighted word list); each song is assigned its strongest theme."""
    n_docs, n_terms = X.shape
    k = min(n_themes, n_docs - 1, n_terms)
    if k < 2:
        return []
    from sklearn.decomposition import NMF
    # solver="mu": the coordinate-descent solver of scikit-learn 1.5.2 raised
    # "ndarray is not C-contiguous" on some inputs (reproduced with the demo data).
    nmf = NMF(n_components=k, init="nndsvda", solver="mu", random_state=0, max_iter=1000)
    X = np.ascontiguousarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)
    W = nmf.fit_transform(X)
    H = nmf.components_
    out = []
    for c in range(k):
        words = [terms[j] for j in np.argsort(-H[c])[:8] if H[c, j] > 0]
        members = [songs[i].title for i in range(n_docs) if W[i].argmax() == c and W[i, c] > 0]
        out.append({"theme": c + 1, "words": words, "songs": members, "share": len(members) / n_docs})
    return [t for t in out if t["songs"]]


def similarity_matrix(report: LyricsReport) -> Optional[np.ndarray]:
    """Cosine similarity between songs' lyrics (TF-IDF rows are already L2-normalised)."""
    if report.tfidf_matrix is None:
        return None
    return report.tfidf_matrix @ report.tfidf_matrix.T


def entropy_bits(counter: Counter) -> float:
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0
