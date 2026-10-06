import streamlit as st
import rag

st.title("RAG powered by Jev")
question = st.text_input("Ask a question")

if question:
    passages, p, kept, answer = rag.ask(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Jev score for each chunk")
    for cid, text in passages.items():
        st.progress(p[cid], text=f"{cid}: {p[cid]:.2f} {'kept' if cid in kept else 'dropped'}")

    st.write(f"Answerable: {p['answerable']:.2f}  |  Injection: {p['injection']:.2f}")
