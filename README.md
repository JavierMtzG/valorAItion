# 🌟 ValorAItion - Análisis de Opiniones 🌟



ValorAItion es un proyecto de análisis de opiniones basado en aprendizaje automático. Utiliza técnicas de procesamiento de lenguaje natural (NLP) para clasificar las opiniones de los usuarios en positivas o negativas. 🎉

## 📂 Estructura del Proyecto
```plaintext
valorAItion/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── preprocess.py
│   └── model.py
├── utilities/
│   └── text_utilities.py
│   └── stop_words_english.txt
├── train_model.py
├── predict.py
├── evaluate.py
├── LICENSE
├── .gitignore
├── README.md
└── .git
```

## 🚀 Comenzando
### Prerrequisitos
- Python 3.x
### Instalación
**Clonar el repositorio**:
```bash
git clone https://github.com/JavierMtzG/valorAItion.git
cd valorAItion
```
**Crear un entorno virtual**:

```
python -m venv venv
```
**Activar el entorno virtual**:
```
.\venv\Scripts\activate #ó source venv/bin/activate
```
**Instalar las dependencias**:
```
pip install -r requirements.txt
```

## Datos
Asegúrate de que los archivos train.csv y test.csv estén en la carpeta data. Estos archivos deben contener las columnas polarity, title y text.

### Entrenamiento del Modelo
Para entrenar el modelo, ejecuta:
``python train_model.py``

### Evaluación del Modelo
Para evaluar el modelo, ejecuta:
``python3 evaluate.py``
# Los resultados se guardarán en evaluation_results.csv, confusion_matrix.csv y classification_report.csv.

### Predicción de Opiniones
Para predecir la polaridad de una nueva opinión, ejecuta:

``python3 predict.py``

Introduce el título y el texto de la opinión cuando se te solicite.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE.

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Por favor, abre un issue o un pull request para cualquier cambio o sugerencia.

## 📧 Contacto
Si tienes alguna pregunta, por favor contacta a javier.mgarcia29@gmail.com
