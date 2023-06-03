# **Codefest-gamma-ai**

# **Installation**

First of all, be sure to conda environment with Python 3.8 installed. To do use, execute the command:
```
conda create --name codefest-test python=3.8  
```

The installation consist of two simple steps: 

Build the library

```python
python setup.py bdist_wheel
```

Install the library:

```python
pip install dist/gamma-0.1.0-py3-none-any.whl
```

## **Task 1: Text**

In this task the user is able to peform token and text classifications related to 
the Amazon forest news articles. The main functionality of this part of the library consists of 
Named Entity Recognition from the classes Person, Organization, Location and Miscellaneous, and also Text classification in order to asses the impact of the news article, the possible impact types are Mining, Contamination, Deforestation and None.

On the first part, in order to perform the **NER** task we decided to use the [Flair](https://github.com/flairNLP/flair) [`flair/ner-spanish-large`](https://huggingface.co/flair/ner-spanish-large) Spanish NER model directly. This model is one of the best models available for the NER task in Spanish. This is a transformer based model, so it's best used when there is **GPU accelerated** hardware available. 

For the second task that is **text classification**, we opted to use the [`Recognai/bert-base-spanish-wwm-cased-xnli`](https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli) transformer based model from HuggingFace in the modality of zero shot classification, which proved to give decent results. As well as the NER model, the text classification model is best used if there is **GPU accelerated** hardware available.

Now we show a few short examples of the library, especifically for the `ner_from_str` method:

```python
from gamma import text

text = """El aumento de las represas en la Amazonia amenaza el flujo natural de sus ríos, altera los ciclos naturales
y pone en grave riesgo especies como los delfines y peces migratorios. El suministro de agua para las comunidades locales y el transporte de
alimentos, también se ven afectados por cuenta de la producción de energía en la selva amazónica. En la Amazonia hay 154 represas
para la producción de energía hidroeléctrica y se planea la construcción de otras 277 en los próximos años."""

text.ner_from_str(text, output_path="entities.json")
## output JSON
# {'text': 'El aumento de las represas en la Amazonia amenaza el flujo 
#           ... construcción de otras 277 en los próximos años.',
#  'org': [],
#  'loc': ['Amazonia'],
#  'per': [],
#  'misc': [],
#  'impact': 'Contaminacion'}
```

Also an example from `ner_from_url`:
```python
from gamma import text
url = "https://elmorichal.com/mineria-ilegal-radiografia-del-rio-inirida/"
text.ner_from_url(url, output_path="entities.json")
## output JSON
# {'text': 'Read Time: 7 Minute, 21 Second\n\nEl Guainía es sinónimo de agua y vida, siendo uno de los departamentos con mayor presencia de fuentes hídricas dulces de Colombia, que en conjunto con las densas selvas que se levantan sobre su territorio, la convierte en el hogar de miles de especies de fauna y flora, que aporta una gran proporción del oxígeno para un país, que cada día se ahoga en el efecto invernadero.\n\nUna muestra de esta magna riqueza es la Estrella Fluvial de Inírida, una imponente maravilla ambiental donde convergen los ríos Inírida, Guaviare y Atabapo, declarado en el año 2014 como sitio Ramsar, luego de comprender la importancia a nivel nacional e internacional que representa la conservación y preservación de estas vertientes de agua dulce y su biodiversidad.\n\nSin embargo, este título otorgado y que trasciende las fronteras del país, no ha sido suficiente para garantizar que un enemigo silencioso y del que poco se habla en la agenda nacional, tome protagonismo en las aguas que se extienden a lo largo del departamento del Guainía: La minería ilegal.\n\nEl río Inírida es uno de los más amenazados, Grupos Armados Organizados (GAO) y Grupos Armados Organizados Residuales (GAO-r) delinquen a lo largo de sus 1.300 kilómetros de cuenca, encontrando en la explotación ilícita de yacimientos mineros un músculo económico, para financiar sus acciones delictivas, aprovechándose de las limitaciones de acceso terrestre, fluvial y aérea que existen para llegar hacia esos territorios remotos en la geografía colombiana.\n\nEl 80% de la población que habita en el departamento del Guainía son comunidades indígenas entre estas: los Curripacos, Puniave, Sikuani y Piapoco, quienes viven de la pesca y la agricultura. Justamente las limitaciones para sacar los productos generados de estas prácticas, incentivó un cambio en la economía de las poblaciones ribereñas, que vieron una opción más rentable en el desarrollo de actividades mineras, empleando maquinarias y químicos proporcionados por los GAO y GAO-r, dejando atrás el barequeo, la forma artesanal en la se emplean bateas para extraer el oro sin causar daños en el medio ambiente.\n\nFoto: Cortesía Prensa Fuerza Naval del Oriente\n\nNo obstante, el panorama y las consecuencias de esta actividad es completamente diferente. La maquinaria es ubicada al interior de balsas flotantes construidas en madera y techos de paja, que durante día y noche se desplazan por las aguas del río Inírida socavando y extrayendo su suelo en la búsqueda de oro aluvial, arrasando sin discriminación con animales y plantas.\n\nDentro del proceso, son empleados químicos como el cianuro y el mercurio para la separación del oro de las piedras y/o sedimentación, en un proceso que se denomina amalgamar, y que sus residuos son vertidos en las corrientes, contaminando sus aguas, generando daños irreparables a los ecosistemas y en la salud de las comunidades que viven del río.\n\nPor su parte, los ingresos recibidos por realizar estas labores son insignificantes en comparación a las extensas y duras jornadas a los que son sometidos en condiciones extremas los trabajadores, quienes basan sus funciones, en la mayoría de los casos, en llevar los «dragones” hasta el suelo del río, enfrentando profundidades entre los 14 y 18 metros. Los dineros adquiridos por la comercialización del oro bruto fortalecen a los Grupos Armados Organizados de forma directa.\n\nEste flagelo liderado por criminales, no solo está fracturando las comunidades indígenas del Guainía, sino que está dejando una deuda ambiental con Colombia y el planeta, al destruir su verdadero tesoro: El Agua.\n\nEl agua: activo estratégico de la nación\n\nColombia ocupa el tercer lugar entre los países con más agua en el mundo, siendo este recurso un activo estratégico de la nación, y uno de los principales y prevalentes intereses nacionales, por tanto su protección y defensa es constituida como prioridad de seguridad nacional, dentro de la Política de Defensa y Seguridad del país.\n\nLas Fuerzas Militares y de Policía, cumplen un rol preponderante en la protección de los recursos naturales. Su compromiso para contrarrestar los delitos que afecten el medio ambiente, en especial la minería ilegal y criminal, se reflejan en el planeamiento y ejecución de operaciones focalizadas a la neutralización de estas actividades que lucran las arcas de las organizaciones al margen de la ley.\n\nLa articulación de capacidades fluviales, terrestres, aéreas y judiciales, son esenciales para el éxito de este tipo de operaciones, debido a la complejidad del terreno donde se presenta este delito en el departamento del Guainía. Un ejemplo del trabajo en equipo, fue el más reciente golpe propinado a este delito por parte de la Fuerza de Tarea contra la Minería.\n\nFuerza de Tarea Fluvial Conjunta y Coordinada ontra la minería\n\nUn operativo sin precedentes contra la minería ilegal en la Orinoquía, fue desarrollado en los últimos días por unidades de la Armada de Colombia de forma conjunta, coordinada e interagencial con el Ejército Nacional, Fuerza Aérea Colombiana, Policía Nacional y la Fiscalía, logrando reducir de forma significativa la contaminación del río Inírida y propinando un golpe contundente a una de las principales fuentes de finanzas de Grupos Armados Organizados Residuales.\n\nPara la ejecución de esta operación fueron empleados doce helicópteros, donde tropas de las Fuerzas Militares y de Policía se desplazaron de forma simultánea hasta los sectores de Punta Tigre, Buenavista y Punta Yuca, ubicados a lo largo del río Inírida, Guainía, cuya información de inteligencia había identificado infraestructuras dedicadas a la extracción ilegal de oro.\n\nFoto: Cortesía Prensa Fuerza Naval del Oriente\n\nUna vez sobre los puntos de interés para dar el mencionado golpe, con limitaciones de combustible debido a las largas distancias recorridas, de forma simultánea los uniformados iniciaron una maniobra de asalto aéreo y fluvial a contrarreloj, desembarcando de las aeronaves seis botes tipo Zodiac, sobre el cauce del río. A bordo de los botes, iniciaron el desplazamiento hasta las zonas donde se hallaban las barcazas de fabricación artesanal, sorprendiendo a quienes desarrollaban la extracción ilícita de yacimientos mineros.\n\nAl notar la presencia de las tropas, estos sujetos emprendieron la huida a través de varias trochas.\n\nEn el registro de las estructuras, fueron hallados maquinarias y elementos empleados para la extracción ilegal de recursos mineros, que registran un valor aproximado de tres mil millones de pesos.\n\nFoto: Cortesía Prensa Fuerza Naval del Oriente\n\nDe acuerdo a labores de inteligencia, estas barcazas ilegales pertenecerían a las redes de finanzas del Grupo Armado Organizado Residual, subestructura Acacio Medina que delinque bajo el mando de Alias Jhon 40, sindicados de controlar la minería ilegal en esta zona del país, siendo responsables de los incalculables daños ambientales ocasionados a esta importante fuente hídrica del departamento del Guainía.\n\nTres dragas, la maquinaria y los elementos hallados en su interior, fueron destruidos de forma controlada, por expertos en explosivos, en el lugar de los hechos.\n\n¡Estamos a tiempo de salvar el río inírida!\n\nLa historia de los ríos Cauca y Sambingo, no se puede repetir en los ríos de la Orinoquía colombiana. No tomar acciones hoy significaría ver al cabo de unos años, la Estrella Fluvial de Inírida convertida en un desierto. Por esto, el resultado dado por la Fuerza de Tarea Fluvial y Coordinada Contra la Minería, envía un mensaje determinante en la lucha frontal contra la explotación ilícita de yacimientos mineros y en la protección de los recursos naturales.\n\nTan solo la destrucción de las tres dragas reduce de forma significativa la contaminación de sus aguas a lo largo de 130 kilómetros, y en general en sus principales afluentes evitando: el vertimiento desaforado de cientos de kilogramos de mercurio y cianuro, los cuales generan problemas en la salud de la población ya sea por consumo del agua o del pescado contaminado, afectando principalmente el sistema nervioso central, el sistema urinario y la visión, en una región donde el acceso a servicios médicos es mínimo.\n\nAsí como la población, la fauna de la región también se encuentra amenazada, especies en peligro crítico de extinción como el Delfín Rosado, La Nutria y El Jaguar, tienen su habitad en el río Inírida. La desaparición de estas especies por efectos de la minería ilegal, representaría para Colombia una pérdida devastadora. Lo cierto, es que cuantificar el daño causado por no actuar a tiempo frente a la minería criminal, es incalculable y el impacto sería irreversible.\n\nEl río Inírida está a tiempo de salvarse y la continua ejecución de operaciones ofensivas por parte de las Fuerza Pública, es una de las medidas de contingencia. Sin embargo, este impulso no es suficiente para confrontar esta problemática, es imprescindible el trabajo interinstitucional adelantado con organizaciones como World Wildlife Fundation (WWF), Corporinoquía, Corpoamazonía, la Corporación para el Desarrollo Sostenible del Norte y del Oriente (CDA), Fundación Orinoquía, Fundación Omacha, entre otras; con las que actualmente se vienen adelantando acercamientos para la estructuración de planes aterrizados en atender el fenómeno de la minería ilegal en el Guainía.\n\nAbout Post Author Redacción ElMorichal edwinesn@hotmail.com\n\nHappy 0 0 % Sad 0 0 % Excited 0 0 % Sleepy 0 0 % Angry 0 0 % Surprise 0 0 %',
#  'org': ['Grupos Armados Organizados',
#   'GAO',
#   'Grupos Armados Organizados Residuales',
#   'GAO-r',
#   'Curripacos',
#   'Puniave',
#   'Sikuani',
#   'Piapoco',
#   'Fuerza Naval del Oriente',
#   'Fuerzas Militares',
#   'Policía',
#   'Fuerza de Tarea contra la Minería',
#   'Fuerza de Tarea Fluvial',
#   'Armada de Colombia',
#   'Ejército Nacional',
#   'Fuerza Aérea Colombiana',
#   'Policía Nacional',
#   'Fiscalía',
#   'Cortesía Prensa',
#   'Grupo Armado Organizado Residual',
#   'Acacio Medina',
#   'Coordinada Contra la Minería',
#   'Fuerza Pública',
#   'World Wildlife Fundation',
#   'WWF',
#   'Corporinoquía',
#   'Corpoamazonía',
#   'Corporación para el Desarrollo Sostenible del Norte y del Oriente',
#   'CDA',
#   'Fundación Orinoquía',
#   'Fundación Omacha',
#   'Redacción',
#   'ElMorichal'],
#  'loc': ['Guainía',
#   'Colombia',
#   'Orinoquía',
#   'río Inírida',
#   'Punta Tigre',
#   'Buenavista',
#   'Punta Yuca'],
#  'per': ['Alias Jhon 40', 'edwinesn'],
#  'misc': ['Estrella Fluvial de Inírida',
#   'Inírida',
#   'Guaviare',
#   'Atabapo',
#   'sitio Ramsar',
#   'El Agua',
#   'Política de Defensa y Seguridad',
#   'Zodiac',
#   'Cauca',
#   'Sambingo',
#   'Orinoquía',
#   'Delfín Rosado',
#   'La Nutria',
#   'El Jaguar'],
#  'impact': 'Mineria'}
```
****

## **Task 2: Object detection**

Here the second task is the object detection on satellital videos from a plane. The objects that are relevant to detect are:
* Buildings. 
* Vehicles.
This is done using the models 
