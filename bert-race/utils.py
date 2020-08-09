import os
import pickle
import random
import sys

import neuralcoref
import nltk
import spacy
from nltk.corpus import names

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

nltk.download('names')

MALE_NAMES = names.words('male.txt')
FEMALE_NAMES = names.words('female.txt')

NAMES = MALE_NAMES + FEMALE_NAMES

ner_dict_path = os.path.join(sys.path[0], 'ner_dict.pkl')
with open(ner_dict_path, 'rb') as f:
    NER_DICT = pickle.load(f)


def get_names(text):
    doc = nlp(text)
    names = []
    for x in doc.ents:
        if x.label_ == 'PERSON':
            names.append(x.text)

    names = list(set(names))
    return names


def get_names_groups(text):
    doc = nlp(text)
    clusters = doc._.coref_clusters
    groups = []
    if clusters:
        for cluster in clusters:
            name_group = get_names('. '.join(list(set([str(i) for i in cluster.mentions]))))

            flag = True
            for name in name_group:
                for group in groups:
                    if name in group:
                        flag = False
                    elif name in ' '.join(group):
                        group.append(name)
                        flag = False

            if name_group and flag:
                groups.append(name_group)

    return groups


def get_entities(text, entity_type):
    doc = nlp(text)
    entities = []
    for x in doc.ents:
        if x.label_ == entity_type:
            entities.append(x.text)

    entities = list(set(entities))
    return entities


def get_adv_names(names_num, name_gender_or_race):
    adv_names = []
    for _ in range(names_num):
        if name_gender_or_race == 'males':
            adv_names.append(random.choice(MALE_NAMES))
        elif name_gender_or_race == 'female':
            adv_names.append(random.choice(FEMALE_NAMES))
        else:
            adv_names.append(random.choice(NAMES))
    return adv_names


def get_adv_entities(entities_num, entity_type):
    adv_entities = []
    for _ in range(entities_num):
        adv_entities.append(random.choice(NER_DICT[entity_type]))
    return adv_entities


def replace_names(text, names, adv_names):
    if not names:
        return text

    for i, name_group in enumerate(names):
        for name in name_group:
            adv_text = text.replace(name, adv_names[i])
            text = adv_text

    return adv_text


def replace_entities(text, entities, adv_entities):
    if not entities:
        return text

    for i, entity in enumerate(entities):
        adv_text = text.replace(entity, adv_entities[i])
        text = adv_text

    return adv_text
