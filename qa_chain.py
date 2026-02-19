"""
qa_chain.py
RAG QA system using FAISS retriever + Ollama (phi3:mini) as local LLM.
Fully compatible with LangChain 1.x modular setup.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# No need to import from langchain.chains in new version
# Retrieval handled manually below.

def ask_question(vectorstore, question: str):
    """Retrieve docs and generate an answer using Ollama."""
    print("🔎 Retrieving relevant documents...")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # ✅ Use retriever.invoke() instead of get_relevant_documents
    docs = retriever.invoke(question)

    # Combine text from top retrieved documents
    context = "\n\n".join([d.page_content for d in docs])

    # 🧠 Create local Ollama model
    llm = ChatOllama(model="phi3:mini", temperature=0)

    # 🗨️ Define how the LLM should answer
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question using the provided context.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    # ⚙️ Chain: prompt → LLM → text output
    chain = prompt | llm | StrOutputParser()

    print("🤖 Generating answer locally using Ollama...")
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs


if __name__ == "__main__":
    from indexer import load_faiss_index

    print("🔍 Loading FAISS index...")
    vectorstore = load_faiss_index()
    print("✅ FAISS index loaded!")

    while True:
        question = input("\nAsk your question (or type 'exit'): ")

    # ✅ Check for exit first
        if question.strip().lower() in ["exit", "quit"]:
            print("Exiting...")
            break

    # Only call the model if it's a real question
        answer, docs = ask_question(vectorstore, question)

        print("\n🧠 Answer:\n", answer)
        print("\n📚 Sources:")
        for i, d in enumerate(docs, 1):
            print(f"Source {i}: {d.metadata}")
            print(d.page_content[:300])
