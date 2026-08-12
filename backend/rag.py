import json
import os
import re


class BankingRAG:

    def __init__(self, data_path):

        self.data_path = data_path
        self.documents = []

        self.load_data()

        print(
            f"RAG loaded {len(self.documents)} documents"
        )

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

        try:

            data = json.loads(content)

            if isinstance(data, list):
                records = data

            else:
                records = [data]

        except json.JSONDecodeError:

            records = []

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

        for record in records:

            self.documents.append({

                "user_query":
                    record.get(
                        "user_query",
                        ""
                    ),

                "combined_completion":
                    record.get(
                        "combined_completion",
                        ""
                    ),

                "answer_guidance":
                    record.get(
                        "answer_guidance",
                        ""
                    ),

                "domain_category":
                    record.get(
                        "domain_category",
                        ""
                    ),

                "subdomain":
                    record.get(
                        "subdomain",
                        ""
                    ),

                "original_context":
                    record.get(
                        "original_context",
                        ""
                    )
            })

    # --------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------

    @staticmethod
    def tokenize(text):

        return set(
            re.findall(
                r"[a-zA-Z0-9]+",
                text.lower()
            )
        )

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=5
    ):

        query_words = self.tokenize(
            query
        )

        if not query_words:
            return []

        results = []

        for document in self.documents:

            searchable_text = " ".join([
                document["user_query"],
                document["answer_guidance"],
                document["original_context"],
                document["domain_category"],
                document["subdomain"]
            ])

            document_words = self.tokenize(
                searchable_text
            )

            common_words = (
                query_words &
                document_words
            )

            if not common_words:
                continue

            score = (
                len(common_words)
                /
                len(query_words)
            )

            result = document.copy()

            result["score"] = score

            results.append(result)

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]
