```python

```


```python
path = '/home/turma02/Imagens/dc_small/'
```


```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
```


```python
modelo = ResNet50(weights = 'imagenet')
```


```python
img = path + 'images.jpeg'
```


```python
img  = image.load_img(img, target_size=(224, 224))
```


```python
plt.imshow(img)
```


```python
img_p = image.img_to_array(img)
img_p = np.expand_dims(img_p, axis = 0)
img_p = preprocess_input(img_p)
```


```python
preds = modelo.predict(img_p)
```


```python
print('Predições: ', decode_predictions(preds, top = 3)[0])
```


```python
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
```


```python
num_classes = 2
```


```python
tf.random.set_seed(0)
```


```python
modelo_base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
```


```python
for layer in modelo_base.layers:
    layer.trainable = False
```


```python
modelo = tf.keras.Sequential(
    [modelo_base,
     GlobalAveragePooling2D(),
     Dense(1024, activation = 'relu'),
     Dense(num_classes, activation = 'softmax')
    ])
```


```python
path = '/home/turma02/Imagens/dc_small/'
```


```python
def cria_dataframe(caminho):
    arquivos = []
    rotulos = []
    for arquivo in os.listdir(caminho):
        if arquivo.startswith('cat'):  # Corrigido para acessar o nome do arquivo
            rotulos.append('cat')
        elif arquivo.startswith('dog'):  # Corrigido para acessar o nome do arquivo
            rotulos.append('dog')
        arquivos.append(arquivo)
    return pd.DataFrame({'arquivo': arquivos, 'rotulo': rotulos}) 
```


```python
# Criando o DataFrame a partir do diretório de treinamento
df_dados = cria_dataframe(os.path.join(path, 'train'))
```


```python
df_dados.info()
```


```python
# Particionando a base com nova proporção
train_df, validation_df = train_test_split(df_dados, test_size=0.3, stratify=df_dados['rotulo'], random_state=0)

```


```python
test_df = cria_dataframe(os.path.join(path, 'test'))
```


```python
# Data Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
```


```python
# Criando os geradores
def create_generator(datagen, dataframe, directory):
    return datagen.flow_from_dataframe(
        dataframe,
        directory=directory,
        x_col='arquivo',
        y_col='rotulo',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

train_generator = create_generator(train_datagen, train_df, os.path.join(path, 'train'))
validation_generator = create_generator(validation_datagen, validation_df, os.path.join(path, 'train'))
test_generator = create_generator(test_datagen, test_df, os.path.join(path, 'test'))
```


```python
# Compilando o modelo com variação de otimizador
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
```


```python
# Treinando o modelo com número ajustado de épocas
modelo.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)
```


```python
# Avaliando no conjunto de teste
eval_result = modelo.evaluate(test_generator)
print(f"Acurácia no conjunto de teste: {eval_result[1] * 100:.2f}%")
```


```python
test_loss, test_accuracy = modelo.evaluate(test_generator, steps = test_generator.samples // test_generator.batch_size)
print(f'Acuracia no Teste: {test_accuracy}')
```


```python
modelo.save('resnet_10_epoch_dogs_cats.keras')
```


```python
def predicao_img(imagem, modelo):
    img = image.load_img(imagem, target_size = (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = preprocess_input(img_array)
    predicoes = modelo.predict(img_array)
    classes = {0: 'cat', 1: 'dog'}
    classes_predita = classes[np.argmax(predicoes[0])]
    return classes_predita, img
```


```python
amostra = path + 'img9.jpeg'
classe_predita, img_resposta = predicao_img(amostra, modelo)
plt.figure(figsize = (3, 3))
plt.imshow(img_resposta)
plt.title(f'Classe: {classe_predita}')
plt.show()
```


```python

```


```python

```


```python

```


```python

```


```python

```
