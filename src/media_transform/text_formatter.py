import pandas as pd
import spacy
from tqdm.auto import tqdm
from spacy.lang.es import Spanish


def text_formatter(recording_text: str) -> str:
    # Remove crutches when speaking
    recording_text = recording_text.replace('eh?', '.').replace('eh', '.')
    return recording_text


def sentence_formatter(sentence_text: str):
    nlp = spacy.load('en_core_web_sm')
    nlp = Spanish()
    nlp.add_pipe('sentencizer')

    doc = nlp(sentence_text)

    return list(doc.sents)


def read_paragraphs(paragraph: str) -> str:
    pass


def preprocess_recording_transcription(recording_transcrip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the recording transcription dataframe.
    :param recording_transcrip_df: recording transcription dataframe
    :return: recording transcription dataframe processed
    """
    recording_transcrip_df['Transcript'] = recording_transcrip_df['Transcript'].apply(text_formatter)
    recording_transcrip_df['SentenceCharCount'] = recording_transcrip_df['Transcript'].apply(lambda x: len(x))
    recording_transcrip_df['SentenceWordCount'] = recording_transcrip_df['Transcript'].apply(lambda x: len(x.split(" ")))
    recording_transcrip_df['SentenceCountRaw'] = recording_transcrip_df['Transcript'].apply(lambda x: len(x.split(". ")))
    recording_transcrip_df['TranscriptSentences'] = recording_transcrip_df['Transcript'].apply(sentence_formatter)

    for general_sentence_index, sentence in tqdm(enumerate(recording_transcrip_df.iterrows())):
        index_sentence = sentence[0]
        sentence_data = sentence[1]

    return recording_transcrip_df
