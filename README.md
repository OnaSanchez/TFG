# Estudi pilot per a la mesura de l'estrès tèrmic

El fenomen dels efectes tèrmics ha emergit
com una qüestió d'importància creixent en un
món caracteritzat per la variabilitat climàtica i
l'exposició a condicions ambientals extremes. Aquest
fenomen cobra una rellevància extraordinària en
el context actual, en el qual el canvi climàtic i els
avenços en la indústria han elevat l'exposició de
les persones a condicions d'estrès tèrmic.

L’objectiu principal d’aquest projecte consisteix
en proposar una mesura d’estrès tèrmic basada en
senyals electrofisiològiques, objectiva, no invasiva
i fàcil d’aplicar.

## Materials produïts
Dins de la carpeta TFG, es troben diversos fitxers i carpetes necessaris per a la realització de l'estudi.

- main.py: És l'arxiu principal que s'ha de executar. Conté crides a funcions essencials per a la creació de la base de dades, realitza la neteja de les dades proporcionades i executa l'anàlisi, guardant els resultats.

- Carpeta database: Conté l'arxiu mongo.py, responsable de la interacció amb la base de dades MongoDB. Les funcions d'aquest fitxer es dediquen a la creació de les col·leccions 'Usuari', 'Senyal', 'Mesura', 'Test_psico' i 'Sessió' a una base de dades MongoDB, afegint les dades corresponents.

- Carpeta data_analysis: Conté l'arxiu analysis.py, que inclou funcions per a l'estudi de les dades, implementa models de regressió i genera gràfics per facilitar la visualització de l'estructura de les dades i el seu anàlisi. També s'hi troba l'arxiu dataset.py, format per funcions relacionades amb la preparació i neteja del conjunt de dades per a l'anàlisi.

Aquesta estructura separa clarament les funcions relacionades amb el processament de dades, la connectivitat a la base de dades i l'anàlisi estadística, dividint el projecte en tasques específiques.

## Autora

- Ona Sánchez Núñez, NIU: 1601181
