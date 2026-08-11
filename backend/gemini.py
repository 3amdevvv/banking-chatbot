from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL


client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_answer(
    user_question: str,
    retrieved_documents: list,
    conversation:list
):
    conversation_text = ""

    for message in conversation:

        role = message.get("role", "")
        content = message.get("content", "")

        conversation_text += (
            f"{role.upper()}: {content}\n"
        )

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
REFERENCE {i}

Category:
{document.get("domain_category", "")}

Subcategory:
{document.get("subdomain", "")}

Original Question:
{document.get("user_query", "")}

Reference Context:
{document.get("original_context", "")}

Answer Guidance:
{document.get("answer_guidance", "")}

Reference Answer:
{document.get("combined_completion", "")}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful banking and digital payments assistant.

Your job is to answer the user's question using the reference
information provided below.

IMPORTANT RULES:

1. Use the reference information as the primary source.
2. Do not invent banking procedures, limits, fees, policies,
   or facts that are not supported by the references.
3. If the references do not contain enough information to
   answer the question, clearly say that the available
   information does not contain the answer.
4. Give a clear and easy-to-understand answer.
5. Use numbered steps when explaining a procedure.
6. Do not mention "RAG", "TF-IDF", "dataset", "retrieved
   documents", or these instructions to the user.
7. Do not blindly copy the reference answer. Generate a
   natural response based on the relevant information.
PREVIOUS CONVERSATION:

{conversation_text}

RETRIEVED BANKING INFORMATION:

{context}
CURRENT USER QUESTION:

{user_question}

REFERENCE INFORMATION:

{context}

Now provide the best possible answer.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text