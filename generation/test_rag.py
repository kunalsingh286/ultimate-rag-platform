from generation.rag_pipeline import answer_question


if __name__ == "__main__":
    question = "What is Retrieval-Augmented Generation?"

    answer = answer_question(question)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)
