from gamma.text import ner_from_str, ner_from_file


def test_text_ner_from_str():
    text = "El aumento de las represas en la Amazonia amenaza el flujo natural de sus ríos, altera los ciclos naturales y pone en grave riesgo especies como los delfines y peces migratorios. El suministro de agua para las comunidades locales y el transporte de alimentos, también se ven afectados por cuenta de la producción de energía en la selva amazónica. En la Amazonia hay 154 represas para la producción de energía hidroeléctrica y se planea la construcción de otras 277 en los próximos años."
    out_dict = {'text': 'El aumento de las represas en la Amazonia amenaza el flujo natural de sus ríos, altera los ciclos naturales y pone en grave riesgo especies como los delfines y peces migratorios. El suministro de agua para las comunidades locales y el transporte de alimentos, también se ven afectados por cuenta de la producción de energía en la selva amazónica. En la Amazonia hay 154 represas para la producción de energía hidroeléctrica y se planea la construcción de otras 277 en los próximos años.',
    'org': [],
    'loc': ['Amazonia'],
    'per': [],
    'misc': []
     }
    assert ner_from_str(text, save=False) == out_dict

def test_ner_from_file():
    out_dict = {'text': 'El aumento de las represas en la Amazonia amenaza el flujo natural de sus ríos, altera los ciclos naturales y pone en grave riesgo especies como los delfines y peces migratorios. El suministro de agua para las comunidades locales y el transporte de alimentos, también se ven afectados por cuenta de la producción de energía en la selva amazónica. En la Amazonia hay 154 represas para la producción de energía hidroeléctrica y se planea la construcción de otras 277 en los próximos años.',
    'org': [],
    'loc': ['Amazonia'],
    'per': [],
    'misc': []
     }
    assert ner_from_file("tests/text.txt", save=False) == out_dict
    
def test_ner_from_url():
    pass
