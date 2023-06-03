import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from newspaper import Article
import json
import transformers

# load tagger
tagger = SequenceTagger.load("flair/ner-spanish-large")
# load zero shot classifier
classifier = transformers.pipeline("zero-shot-classification", 
                   model="Recognai/bert-base-spanish-wwm-cased-xnli")

def save_to_json(json_dict, output_path):
    with open(output_path, "w") as fp:
        json.dump(json_dict,fp)

def classify_text(text):
    pred = classifier(
            text,
            candidate_labels=[
                "Deforestación",
                "Mineria",
                "Contaminacion",
                "Ninguna"
            ],    
            hypothesis_template="Esta noticia trata de {}."
    )
    scores = pred["scores"]
    max_prob = scores.index(max(scores))

    return pred["labels"][max_prob]

def ner_from_str(text, output_path="entities.json", save=True):
    out = {"text": text, "org": [], "loc": [], "per": [], "misc": []}
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    for entity in sentence.get_spans('ner'):
        tag = entity.text
        if tag not in out[entity.tag.lower()]:
            out[entity.tag.lower()].append(tag)
    
    impact = classify_text(text)
    out["impact"] = impact

    if save:
        save_to_json(out, output_path)
        
    return out

def ner_from_file(text_path, output_path="entities.json", save=True):
    text = open(text_path, "r").read()
    return ner_from_str(text, output_path, save)
    
def ner_from_url(url, output_path="entities.json", save=True):
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    
    return ner_from_str(text, output_path, save)
