"""Calculate Table 3's pairwise measurements from the generated response text."""

import json
import re

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from reproduce import (
    DATA_DIR, MODELS, PROCESSED_DATA_DIR, PROJECT_DIR,
    EMBEDDING_MODEL, MODEL_REVISION, EMBEDDING_MAX_TOKENS,
    EMBEDDING_BATCH_SIZE, EMBEDDING_CHUNK_SIZE,
    SPACY_MODEL, SPACY_DISABLED_COMPONENTS, NER_BATCH_SIZE, NER_PROCESSES,
    USE_FULL_ANSWER, LOWERCASE_ENTITIES, EMPTY_SET_OVERLAP,
    FULL_ANSWER_PATTERN, CITATION_PATTERN, NUMBER_PATTERN,
)


def jaccard(first, second):
    """The data include empty sets; use zero for their overlap."""
    return len(first & second) / len(first | second) if first | second else EMPTY_SET_OVERLAP


def citations(text):
    """Extract article IDs from [1], [1, 2], and [1-3] citation formats."""
    ids = set()
    for group in CITATION_PATTERN.findall(text):
        for item in group.split(","):
            bounds = re.split(r"[-–]", item.strip())
            ids.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return ids


def response_pairs(model, label):
    rows = []
    for condition, folder, filename, fields in [
        ("Baseline", "baseline_responses", "positive_question_responses.json", ["positive1_answer", "positive2_answer"]),
        ("Framed", "responses", "question_responses.json", ["positive_answer", "negative_answer"]),
    ]:
        records = json.loads((DATA_DIR / folder / model / filename).read_text())
        for record in records:
            for template, pair in record["ModelGeneratedAnswersWithQuestions"].items():
                first, second = (pair[field] for field in fields)
                # A multi-turn response contains four answers, each evaluated separately.
                first = first if template == "multiturn" else [first]
                second = second if template == "multiturn" else [second]
                for turn, (a, b) in enumerate(zip(first, second)):
                    rows.append({
                        "model": label, "condition": condition, "review": record["ReviewID"],
                        "template": f"multiturn-{turn}" if template == "multiturn" else template,
                        # Use the requested answer rather than the separate rationale.
                        # Some responses have no section marker and are already plain answers.
                        "first": FULL_ANSWER_PATTERN.split(a)[-1].strip() if USE_FULL_ANSWER else a.strip(),
                        "second": FULL_ANSWER_PATTERN.split(b)[-1].strip() if USE_FULL_ANSWER else b.strip(),
                    })
    return pd.DataFrame(rows)


def main():
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL, revision=MODEL_REVISION,
                                  cache_folder=str(PROJECT_DIR / "tmp" / "huggingface" / "hub"))
    encoder.max_seq_length = EMBEDDING_MAX_TOKENS
    # The paper specifies spaCy but not its model; use its small English NER model.
    nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLED_COMPONENTS)
    output = PROCESSED_DATA_DIR / "response_metrics.csv"
    for model_number, (model, label) in enumerate(MODELS.items()):
        frame = response_pairs(model, label)
        texts = frame[["first", "second"]].to_numpy().ravel().tolist()
        print(f"{label}: extracting entities from {len(texts)} answers", flush=True)
        entities = [{(ent.text.lower() if LOWERCASE_ENTITIES else ent.text).strip() for ent in doc.ents}
                    for doc in nlp.pipe(texts, batch_size=NER_BATCH_SIZE, n_process=NER_PROCESSES)]
        refs = [citations(text) for text in texts]
        # Article IDs are citations, so exclude their brackets before extracting numbers.
        numbers = [set(NUMBER_PATTERN.findall(CITATION_PATTERN.sub("", text))) for text in texts]
        embeddings = []
        for start in range(0, len(texts), EMBEDDING_CHUNK_SIZE):
            embeddings.append(encoder.encode(texts[start:start+EMBEDDING_CHUNK_SIZE], batch_size=EMBEDDING_BATCH_SIZE,
                                             normalize_embeddings=True, show_progress_bar=False))
            print(f"{label}: embedded {min(start+EMBEDDING_CHUNK_SIZE, len(texts))}/{len(texts)} answers", flush=True)
        embeddings = np.concatenate(embeddings)
        # A dot product of normalized embeddings is their cosine similarity.
        frame["cosine"] = np.einsum("ij,ij->i", embeddings[::2], embeddings[1::2])
        for metric, values in [("entity", entities), ("citation", refs), ("numerical", numbers)]:
            frame[metric] = [jaccard(a, b) for a, b in zip(values[::2], values[1::2])]
        for metric, values in [("citation", refs), ("numerical", numbers)]:
            frame[f"empty_{metric}"] = [not (a or b) for a, b in zip(values[::2], values[1::2])]
        frame.drop(columns=["first", "second"]).to_csv(output, index=False,
            mode="w" if model_number == 0 else "a", header=model_number == 0)
        print(f"{label}: saved {len(frame)} independently calculated pairs", flush=True)


if __name__ == "__main__":
    main()
