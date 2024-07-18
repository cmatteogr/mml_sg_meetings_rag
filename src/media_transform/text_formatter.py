import pandas as pd
import spacy
from tqdm.auto import tqdm
import functools
import torch
import numpy as np
#%%
from sentence_transformers import util, SentenceTransformer
import re


def text_formatter(recording_text: str) -> str:
    # Remove crutches when speaking
    recording_text = recording_text.replace('eh?', '.').replace('eh', '.')
    return recording_text


def sentence_formatter(sentence_text: str):
    nlp = spacy.load('en_core_web_sm')
    # nlp = Spanish()
    nlp.add_pipe('sentencizer')
    sentences = nlp(sentence_text).sents
    # Cast sentences to str
    sentences = list(map(lambda s: str(s), sentences))
    return sentences


# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def preprocess_recording_transcription(recording_transcr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the recording transcription dataframe.
    :param recording_transcr_df: recording transcription dataframe
    :return: recording transcription dataframe processed
    """
    recording_transcr_df['Transcript'] = recording_transcr_df['Transcript'].apply(text_formatter)
    recording_transcr_df['SentenceCharCount'] = recording_transcr_df['Transcript'].apply(lambda t: len(t))
    recording_transcr_df['SentenceWordCount'] = recording_transcr_df['Transcript'].apply(lambda t: len(t.split(" ")))
    recording_transcr_df['SentenceCountRaw'] = recording_transcr_df['Transcript'].apply(lambda t: len(t.split(". ")))
    recording_transcr_df['TranscriptSentences'] = recording_transcr_df['Transcript'].apply(sentence_formatter)
    recording_transcr_df['TranscriptSentencesCount'] = recording_transcr_df['TranscriptSentences'].apply(
        lambda ts: len(ts))

    recording_transcr_data = recording_transcr_df.to_dict('records')

    # Chucking sentences
    num_sentence_chunk_size = 10
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(recording_transcr_data):
        item["TranscriptSentencesChunks"] = split_list(input_list=item["TranscriptSentences"],
                                                       slice_size=num_sentence_chunk_size)
        item["TranscriptSentencesChunksNumChunks"] = len(item["TranscriptSentencesChunks"])

    # Split each chunk into its own item
    pages_and_chunks = []
    for item in tqdm(recording_transcr_data):
        for sentence_chunk in item["TranscriptSentencesChunks"]:
            chunk_dict = {}
            chunk_dict["Start time"] = item["Start time"]
            chunk_dict["End time"] = item["End time"]
            chunk_dict["Confidence"] = item["Confidence"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                           joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)

    # FJFJFJ
    df = pd.DataFrame(pages_and_chunks)
    df.describe().round(2)

    # Show random chunks with under 30 tokens in length
    min_token_length = 30
    for row in df[df["chunk_token_count"] <= min_token_length].iterrows():
        print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

    # Create embeddings one by one on the GPU
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    embedding_model.to("cpu")
    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

    # Embed all texts in batches
    text_chunk_embeddings = embedding_model.encode(text_chunks,
                                                   batch_size=32,
                                                   convert_to_tensor=True)

    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import texts and embedding df
    text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(
        device)

    # 1. Define the query
    # Note: This could be anything. But since we're working with a nutrition textbook, we'll stick with nutrition-based queries.
    query = "Medellin"
    print(f"Query: {query}")

    # 2. Embed the query to the same numerical space as the text examples
    # Note: It's important to embed your query with the same model you embedded your examples with.
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # 3. Get similarity scores with the dot product (we'll time this for fun)
    from time import perf_counter as timer

    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    end_time = timer()

    print(f"Time take to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    larger_embeddings = torch.randn(100 * embeddings.shape[0], 768).to(device)
    print(f"Embeddings shape: {larger_embeddings.shape}")

    # Perform dot product across 168,000 embeddings
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=larger_embeddings)[0]
    end_time = timer()

    print(f"Time take to get scores on {len(larger_embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    # 4. Get the top-k results (we'll keep this to 5)
    top_results_dot_product = torch.topk(dot_scores, k=5)

    import textwrap

    def print_wrapped(text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)

    print(f"Query: '{query}'\n")
    print("Results:")
    # Loop through zipped together scores and indicies from torch.topk
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further (and check the results)
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")

    return recording_transcr_df
