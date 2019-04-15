from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image


classificador = Sequential()
#primeira camada CNN
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
#Segunda camada CNN
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

#Camada densa oculta
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
#Camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#Classificardor
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
#Carrega e normaliza imagens 
#Cria lotes de iamgens
gerador_treinamento = ImageDataGenerator(rescale = 1./255)
gerador_teste = ImageDataGenerator(rescale = 1./255)

#diretorio treinamento
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
#diretorio teste
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

#steps_per_epoch -- numero de imagens processadas treinamento
#validation_steps -- numero de imagens processadas teste

history = classificador.fit_generator(base_treinamento, steps_per_epoch = 903,
                            epochs = 10, validation_data = base_teste,
                            validation_steps = 300)
#historico modelo
print(history.history.keys())

# Historico de acuracia 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('modelo acurácia')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# historico de loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('modelo loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()







base_treinamento.class_indices
# {'NOSMILE': 0, 'SMILE': 1}
'''
Teste com nova amostra
'''
teste_nova_amostra = image.load_img('dataset/amostra_teste/Ai_Sugiyama_0001.ppm',
                              target_size = (64,64))
teste_nova_amostra = image.img_to_array(teste_nova_amostra)
teste_nova_amostra /= 255
teste_nova_amostra = np.expand_dims(teste_nova_amostra, axis = 0)
previsao = classificador.predict(teste_nova_amostra)
previsao = (previsao > 0.5)
# {'NOSMILE': False, 'SMILE': True}





