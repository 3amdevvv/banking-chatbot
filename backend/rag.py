import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BankingRAG:

    def __init__(self, data_path: str):

        self.data_path = data_path

        self.documents = []
        self.vectorizer = None
        self.document_vectors = None

        self.load_data()
        self.create_index()

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    def load_data(self):

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        with open(self.data_path, "r", encoding="utf-8") as file:

            content = file.read().strip()

        # Your uploaded file is JSON Lines rather than
        # one normal JSON array.

        records = []

        try:
            # Try normal JSON first
            data = json.loads(content)

            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]

        except json.JSONDecodeError:

            # Fall back to JSONL
            for line in content.splitlines():

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)

                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(records)} records")

        for record in records:

            searchable_text = record.get(
                "row_searchable_text",
                ""
            )

            # If row_searchable_text doesn't exist,
            # construct searchable text from available fields.

            if not searchable_text:

                searchable_text = " ".join([
                    str(record.get("user_query", "")),
                    str(record.get("original_context", "")),
                    str(record.get("answer_guidance", "")),
                    str(record.get("combined_completion", "")),
                    str(record.get("domain_category", "")),
                    str(record.get("subdomain", ""))
                ])

            self.documents.append({
                "user_query": record.get("user_query", ""),
                "combined_completion": record.get(
                    "combined_completion", ""
                ),
                "enhanced_prompt": record.get(
                    "enhanced_prompt", ""
                ),
                "answer_guidance": record.get(
                    "answer_guidance", ""
                ),
                "domain_category": record.get(
                    "domain_category", ""
                ),
                "subdomain": record.get(
                    "subdomain", ""
                ),
                "original_context": record.get(
                    "original_context", ""
                ),
                "searchable_text": searchable_text
            })

    # --------------------------------------------------
    # CREATE TF-IDF INDEX
    # --------------------------------------------------

    def create_index(self):

        if not self.documents:
            raise ValueError("No documents found in dataset.")

        texts = [
            document["searchable_text"]
            for document in self.documents
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )

        self.document_vectors = self.vectorizer.fit_transform(texts)

        print("TF-IDF index created successfully.")

    # --------------------------------------------------
    # CLEAN QUERY
    # --------------------------------------------------

    def clean_query(self, query: str):

        query = query.lower()

        query = re.sub(
            r"[^a-zA-Z0-9₹$% ]",
            " ",
            query
        )

        query = re.sub(
            r"\s+",
            " ",
            query
        )

        return query.strip()

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(self, query: str, top_k: int = 5):

        query = self.clean_query(query)

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarity_scores = cosine_similarity(
            query_vector,
            self.document_vectors
        )[0]

        top_indices = np.argsort(
            similarity_scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            document = self.documents[index].copy()

            document["score"] = float(
                similarity_scores[index]
            )

            results.append(document)

        return results