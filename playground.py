import streamlit as st

def build_contextual_query(user_input, window=3):
    context = ""
    for turn in st.session_state.chat_history[-window:]:
        context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    context += f"User: {user_input}\nAssistant:"
    return context

def update_chat_memory(user_input, model_output):
    st.session_state.chat_history.append({
        "user": user_input,
        "assistant": model_output
    })

def run(llm, embedding_model, vectordb):
    st.title("🎮 Playground")
    st.write("Ask questions using plain English. Get AI-powered answers, grounded in your uploaded documents.")

    # Initialize chat history in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render all previous messages
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])

    # Take new input at bottom
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Show user's message immediately
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("🤖 Thinking... Generating a response based on your query."):
            context_query = build_contextual_query(user_input)

            query_emb = embedding_model.encode(context_query).tolist()
            results = vectordb.query(
                query_embeddings=query_emb,
                n_results=5,
                include=['distances', 'documents', 'metadatas']
            )

            filtered_results = []
            for doc, dist in zip(results['documents'][0], results['distances'][0]):
                similarity = 1 - dist
                if similarity >= -0.5:
                    filtered_results.append(doc)

            if len(filtered_results) == 0:
                response_text = "No relevant information found."
            else:
                augmented_query = f"{filtered_results}\n\n{context_query}"
                response = llm.invoke(augmented_query)
                response_text = response.content

        # Show assistant's reply
        with st.chat_message("assistant"):
            st.write(response_text)

        # Save to chat history
        update_chat_memory(user_input, response_text)