import pandas as pd
import spacy
from tqdm.auto import tqdm
from spacy.lang.es import Spanish
#%%
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
    for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
        print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

    for general_sentence_index, sentence in tqdm(enumerate(recording_transcr_df.iterrows())):
        index_sentence = sentence[0]
        sentence_data = sentence[1]
        print(sentence_data)

    return recording_transcr_df
