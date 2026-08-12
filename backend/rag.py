import json
import os
import re


class BankingRAG:

    def __init__(self, data_path: str):

        self.data_path = data_path
        self.documents = []

        self.load_data()

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    def load_data(self):

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        with open(
            self.data_path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()

        records = []

        # Try normal JSON
        try:

            data = json.loads(content)

            if isinstance(data, list):
                records = data

            elif isinstance(data, dict):
                records = [data]

        except json.JSONDecodeError:

            # JSONL fallback
            for line in content.splitlines():

                line = line.strip()

                if not line:
                    continue

                try:

                    records.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError:
                    continue

        print(
            f"Loaded {len(records)} records"
        )

        # --------------------------------------------------
        # Store only fields required by the chatbot
        # --------------------------------------------------

        for record in records:

            searchable_text = record.get(
                "row_searchable_text",
                ""
            )

            if not searchable_text:

                searchable_text = " ".join([
                    str(record.get(
                        "user_query",
                        ""
                    )),

                    str(record.get(
                        "original_context",
                        ""
                    )),

                    str(record.get(
                        "answer_guidance",
                        ""
                    )),

                    str(record.get(
                        "combined_completion",
                        ""
                    )),

                    str(record.get(
                        "domain_category",
                        ""
                    )),

                    str(record.get(
                        "subdomain",
                        ""
                    ))
                ])

            self.documents.append({
                "user_query": record.get(
                    "user_query",
                    ""
                ),

                "combined_completion": record.get(
                    "combined_completion",
                    ""
                ),

                "answer_guidance": record.get(
                    "answer_guidance",
                    ""
                ),

                "domain_category": record.get(
                    "domain_category",
                    ""
                ),

                "subdomain": record.get(
                    "subdomain",
                    ""
                ),

                "original_context": record.get(
                    "original_context",
                    ""
                ),

                "searchable_text": searchable_text
            })

    # --------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------

    def tokenize(self, text):

        text = text.lower()

        words = re.findall(
            r"[a-zA-Z0-9]+",
            text
        )

        return set(words)

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        query_words = self.tokenize(query)

        if not query_words:
            return []

        scored_documents = []

        for document in self.documents:

            document_words = self.tokenize(
                document["searchable_text"]
            )

            if not document_words:
                continue

            common_words = (
                query_words &
                document_words
            )

            score = (
                len(common_words)
                /
                len(query_words)
            )

            if score > 0:

                result = document.copy()

                result["score"] = score

                scored_documents.append(
                    result
                )

        # Highest score first

        scored_documents.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return scored_documents[:top_k]
