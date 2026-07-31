import pandas as pd, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

meta = pd.read_csv('metaData.csv')
test = pd.read_csv('test.csv')

DOC_MAP = {
    'DPM 2025 Volume I': 'DPM-2025-VOLUME-I.pdf',
    'DPM 2025 Volume II': 'DPM-2025-VOLUME-II.pdf',
    'DFPDS Booklet 2024': 'Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf',
    'Navy Regulations Part I': 'RegsNavyI.pdf',
    'Navy Regulations Part II': 'RegsNavyII.pdf',
    'Navy Regulations Part III': 'RegsNavyIII.pdf',
    'Navy Regulations Part IV': 'RegsNavyIV.pdf',
}
PATTERN = re.compile(r'(DPM 2025 Volume [IV]+|DFPDS Booklet 2024|Navy Regulations Part [IV]+)')

def detect_doc(question):
    m = PATTERN.search(question)
    return DOC_MAP[m.group(1)] if m else None

def clean_text(t):
    t = str(t).replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def is_useful_sentence(s):
    words = s.split()
    if len(words) < 7:
        return False
    alpha_chars = sum(c.isalpha() for c in s)
    if alpha_chars < 0.6 * len(s):
        return False
    # skip lines that are mostly ALL CAPS (titles/headers)
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return False
    return True

meta['clean_text'] = meta['text'].apply(clean_text)

results = []
for _, row in test.iterrows():
    qid = row['id']
    question = row['question']
    doc = detect_doc(question)

    subset = meta[meta['document'] == doc].reset_index(drop=True) if doc else meta

    if len(subset) == 0:
        results.append({'id': qid, 'prediction': 'Not found in provided documents.',
                         'pred_source': doc or 'N/A', 'pred_section': 'N/A'})
        continue

    # Step 1: chunk-level retrieval
    vec1 = TfidfVectorizer(stop_words='english', max_features=8000)
    chunk_matrix = vec1.fit_transform(subset['clean_text'].tolist())
    q_vec1 = vec1.transform([question])
    chunk_sims = cosine_similarity(q_vec1, chunk_matrix)[0]
    top_chunk_idx = np.argsort(chunk_sims)[::-1][:3]  # consider top 3 chunks

    # Step 2: sentence-level retrieval within top chunks
    candidates = []
    for ci in top_chunk_idx:
        chunk_row = subset.iloc[ci]
        sents = re.split(r'(?<=[.!?])\s+', chunk_row['clean_text'])
        for s in sents:
            s = s.strip()
            if is_useful_sentence(s):
                candidates.append({'sentence': s, 'section': chunk_row['section'],
                                    'document': chunk_row['document']})

    if not candidates:
        prediction = subset.iloc[top_chunk_idx[0]]['clean_text'][:300]
        pred_section = subset.iloc[top_chunk_idx[0]]['section']
        pred_source = subset.iloc[top_chunk_idx[0]]['document']
    else:
        cand_texts = [c['sentence'] for c in candidates]
        vec2 = TfidfVectorizer(stop_words='english', max_features=4000)
        cand_matrix = vec2.fit_transform(cand_texts)
        q_vec2 = vec2.transform([question])
        cand_sims = cosine_similarity(q_vec2, cand_matrix)[0]
        best_i = int(np.argmax(cand_sims))
        prediction = candidates[best_i]['sentence']
        pred_section = candidates[best_i]['section']
        pred_source = candidates[best_i]['document']

    results.append({'id': qid, 'prediction': prediction,
                     'pred_source': pred_source, 'pred_section': pred_section})

sub = pd.DataFrame(results)
sub.to_csv('submission.csv', index=False, quoting=1)
print(sub.shape)
pd.set_option('display.max_colwidth', 120)
print(sub.head(8).to_string())
