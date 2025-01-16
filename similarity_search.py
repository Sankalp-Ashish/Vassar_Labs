import chromadb
import csv
import streamlit as st
import os

persist_directory = "chroma_db"
try:
    chroma_client = chromadb.PersistentClient(path=persist_directory)
except Exception as e:
    st.error(f"Error initializing ChromaDB: {e}")
    st.stop()

def process_csv_and_add_to_chroma(file_path, collection_name):
    try:
        documents = []
        ids = []
        id_counter = 1
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = csv.reader(file)
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                documents.append(line[1])
                ids.append(str(id_counter))
                id_counter += 1

        if not documents:
            st.warning("CSV file is empty or contains only a header.")
            return

        existing_collections = [col.name for col in chroma_client.list_collections()]
        if collection_name in existing_collections:
            collection = chroma_client.get_collection(name=collection_name)
            collection.delete()
            collection = chroma_client.create_collection(name=collection_name)
        else:
            collection = chroma_client.create_collection(name=collection_name)

        batch_size = 10
        for doc_batch, id_batch in zip(split_into_batches(documents, batch_size), split_into_batches(ids, batch_size)):
            collection.add(documents=doc_batch, ids=id_batch)
        st.success(f"Successfully added {len(documents)} items to collection '{collection_name}'.")

    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"Error processing CSV or adding to ChromaDB: {e}")

def split_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

st.title("ChromaDB Query Interface")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

default_filename = "menu_items.csv"
collection_name = "my_collection" #Default collection name
if os.path.exists(default_filename) and not st.session_state.data_loaded:
    st.write(f"Using default file: {default_filename}")
    file_path = default_filename
    process_csv_and_add_to_chroma(file_path, collection_name)
    st.session_state.data_loaded = True

elif not st.session_state.data_loaded:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        file_path = "uploaded_file.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("Process CSV and Add to ChromaDB"):
            process_csv_and_add_to_chroma(file_path, collection_name)
            st.session_state.data_loaded = True

if st.session_state.data_loaded:
    query_text = st.text_input("Enter your query", "")
    n_results = st.slider("Number of results", 1, 10, 2)
    if st.button("Query ChromaDB"):
        if not query_text:
            st.warning("Please enter a query.")
        else:
            try:
                existing_collections = [col.name for col in chroma_client.list_collections()]
                if collection_name not in existing_collections:
                    st.warning(f"Collection '{collection_name}' does not exist.")
                else:
                    collection = chroma_client.get_collection(name=collection_name)
                    results = collection.query(query_texts=[query_text], n_results=n_results, include=['documents'])
                    st.write("Query Results:")
                    if results and results['documents']:
                        for i, doc in enumerate(results['documents'][0]):
                            st.write(f"Result {i+1}: {doc}")
                    else:
                        st.write("No matching results found.")
            except Exception as e:
                st.error(f"Error querying ChromaDB: {e}")

if os.path.exists("uploaded_file.csv"):
    os.remove("uploaded_file.csv")