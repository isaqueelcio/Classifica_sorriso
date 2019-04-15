# -*- coding: utf-8 -*-

import shutil
import os
import random
#arq = open('SMILE_list.txt', 'r')
arq = open('NONSMILE_list.txt', 'r')

imagens = arq.readlines()
cont = 1
for linha in imagens:
    linha = linha.replace('\n', '')
    shutil.copy2('faces/'+ linha, 'NOSMILE/')
    print(linha)
    print("cont", cont)
    cont= cont + 1
   
arq.close()
print("teste")
##Separa treino e teste NO SMILE

arq = open('NONSMILE_list.txt', 'r')
imagens = arq.readlines()


aleatorios = random.sample(range(1,603), 150)

cont = 1
for i in aleatorios:
    
    linha = imagens[i]
    linha = linha.replace('\n', '')
    shutil.copy2('NOSMILE/'+ linha, 'dataset/test_set/NOSMILE')
    os.remove('NOSMILE/'+linha)
    print("cont", cont)
    cont= cont + 1


arq.close()

##Separa treino e teste SMILE

arq = open('SMILE_list.txt', 'r')
imagens = arq.readlines()


aleatorios = random.sample(range(1,600), 150)

cont = 1
for i in aleatorios:
    
    linha = imagens[i]
    linha = linha.replace('\n', '')
    shutil.copy2('SMILE/'+ linha, 'dataset/test_set/SMILE')
    os.remove('SMILE/'+linha)
    print("cont", cont)
    cont= cont + 1


arq.close()
