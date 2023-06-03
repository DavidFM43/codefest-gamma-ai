import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from newspaper import Article
import json

# load tagger
tagger = SequenceTagger.load("flair/ner-spanish-large")

def save_to_json(json_dict, output_path):
    with open(output_path, "w") as fp:
        json.dump(json_dict,fp)

def ner_from_str(text, output_path="entities.json", save=True):
    out = {"text": text, "org": [], "loc": [], "per": [], "misc": []}
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    for entity in sentence.get_spans('ner'):
        start = entity.start_position
        end = entity.end_position
        tag = entity.text
        out[entity.tag.lower()].append([start, end, tag])
    
    if save:
        save_to_json(out, output_path)
    return out

def ner_from_file(text_path, output_path="entities.json", save=True):
    text = open(text_path, "r").read()
    ner_from_str(text, output_path, save)
    
def ner_from_url(url, output_path="entities.json", save=True):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    ner_from_str(text, output_path, save)
