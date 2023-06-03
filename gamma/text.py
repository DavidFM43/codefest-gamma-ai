import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from newspaper import Article
import json
import transformers

# load tagger
tagger = SequenceTagger.load("flair/ner-spanish-large")
# load zero shot classifier
classifier = transformers.pipeline(
    "zero-shot-classification", model="Recognai/bert-base-spanish-wwm-cased-xnli"
)


def save_to_json(json_dict, output_path):
    """Saves a dictionary to a json file."""
    with open(output_path, "w") as fp:
        json.dump(json_dict, fp)


def classify_text(text):
    """Classifies a text in one of DEFORESTACION, MINERIA, CONTAMINACION, NINGUNA"""
    pred = classifier(
        text,
        candidate_labels=["DEFORESTACION", "MINERIA", "CONTAMINACION", "NINGUNA"],
        hypothesis_template="Esta noticia trata de {}.",
    )
    probs = pred["scores"]
    max_prob = probs.index(max(probs))

    return pred["labels"][max_prob]


def ner_from_str(text, output_path="entities.json", save=True):
    """
    Extracts named entities from a text. The entities are one of Organization,
    Person, Location and Miscellaneous.

    Args:
        text (str): Written text to extract the entities from.
        output_path (str): Path where to save the JSON output file.
        save (bool): Whether to save the dictionary to a JSON file.

    Returns:
        dict:  dictionary that contains the text, entities and impact of the text.
    """

    out = {"text": text, "org": [], "loc": [], "per": [], "misc": []}
    sentence = Sentence(text)
    tagger.predict(sentence)

    for entity in sentence.get_spans("ner"):
        tag = entity.text
        if tag not in out[entity.tag.lower()]:
            out[entity.tag.lower()].append(tag)

    impact = classify_text(text)
    out["impact"] = impact

    if save:
        save_to_json(out, output_path)

    return out


def ner_from_file(text_path, output_path="entities.json", save=True):
    """
    Extracts named entities from a text path. The entities are one of Organization,
    Person, Location and Miscellaneous.

    Args:
        text_path (str): Path of the text file to extract the entities from.
        output_path (str): Path where to save the JSON output file.
        save (bool): Whether to save the dictionary to a JSON file.

    Returns:
        dict:  dictionary that contains the text, entities and impact of the text.
    """
    text = open(text_path, "r").read()
    return ner_from_str(text, output_path, save)


def ner_from_url(url, output_path="entities.json", save=True):
    """
    Extracts named entities from a url. The entities are one of Organization,
    Person, Location and Miscellaneous. The relevant text is extrated from the url
    web page and then the entities are extracted from it.

    Args:
        url (str): Url of the webpage to extract text from which to extract the entities.
        output_path (str): Path where to save the JSON output file.
        save (bool): Whether to save the dictionary to a JSON file.

    Returns:
        dict:  dictionary that contains the text, entities and impact of the text.
    """
    article = Article(url)
    article.download()
    article.parse()
    text = article.text

    return ner_from_str(text, output_path, save)
