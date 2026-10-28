import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = os.path.join("data", "policies")

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
                 "If the answer is not in the context, say that you could not find the "
                 "information in the policy documents.")


def chunk_words(text, size=40, overlap=10):
    """Split text into chunks of `size` words; each new chunk starts size-overlap words later."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):      # this chunk reached the last word
            break
        start += step
    return chunks


def load_chunks(size=40, overlap=10):
    out = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        for c in chunk_words(text, size, overlap):
            out.append({"source": os.path.basename(path), "text": c})
    return out


class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in order]


_default = None

def get_retriever():
    global _default
    if _default is None:
        _default = Retriever(40, 10)
    return _default


def retrieve(query, k=3):
    return get_retriever().retrieve(query, k)


def answer(question, retriever=None):
    r = retriever or get_retriever()
    chunks = r.retrieve(question, 3)
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    user = "CONTEXT:\n" + context + "\nQUESTION:\n" + question
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [c["source"] for c in chunks]