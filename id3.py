import math
import numpy as np
import os
import pandas as pd
import platform
import pprint as pp
import pygame
import random as rand
import tkinter
import tkinter.filedialog

#############################################################
######################## SYS CONFIG #########################
#############################################################


if platform.system() == "Darwin":
  path = "Datasets/"
  assets_path = "Assets/"
else:
  path = "Datasets\\"
  assets_path = "Assets\\"

data_binary = pd.read_csv(path + 'student-mat_csv_binary.csv')
data_erasmus = pd.read_csv(path + 'student-mat_csv_erasmus.csv')


#############################################################
######################### DATA SPLIT ########################
#############################################################

def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data

#############################################################
########################## ENTROPY ##########################
#############################################################

def entropy(target_col):
    """
    Calcula la entropía de un dataset.
    Entrada: la columna a la que queremos calcular la entropía.
    Salida: entropía
    """

    #Trae los elementos únicos, y la cantidad, de una columna
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

#############################################################
######################### INFO GAIN #########################
#############################################################

def InfoGain(data,split_attribute_name,target_name):
    """
    Calcula la ganancia de información de un dataset.
    Entrada: data = dataset
            split_attribute_name = nombre del atributo sobre el cual calcularemos la ganancia de info
            target_name = nombre de la clase
    Salida: ganancia de información
    """

    #Entropía total del dataset
    total_entropy = entropy(data[target_name])


    #Trae los valores únicas y la cuenta del atributo con el cual vamos a dividir el data set
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)

    #Calcula la entropía ponderada
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

    #Calculate the information gain
    info_gain = total_entropy - Weighted_Entropy
    return info_gain

#############################################################
############################ ID3 ############################
#############################################################

# Crea el arbol con el algoritmo ID3
#
# Entrada:
#        data = dataset sobre el cual corre el algoritmo, primero
#               corre en el total del dataset y posteriormente
#               las particiones.
#        originaldata = dataset original
#        features = atributos
#        target_attribute_name = nombre de la clase
#        parent_node_class = Valor o clase del atributo "clase"
#                            del nodo actual
#
# Salida: 1. Regresa el valor de los atributos si estos tienen
#            el mismo valor
#         2. Valor del nodo del atributo en el dataset original
#         3. El valor del atributo del nodo padre
#         4. Árbol

def ID3(data, originaldata, features, target_attribute_name, parent_node_class = None):
    #Criterios para detener el algoritmo, si se satisface algunas de las
    # siguientes condiciones, regresa un nodo hoja

    #Si todos los atributos tienen el mismo valor, regresa este valor
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    #Si el dataset está vacío, regresa el valor del nodo del atributo en el dataset original
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]

    #Si la variable feature es vacía, regresa el valor del atributo del nodo padre
    elif len(features) == 0:
        return parent_node_class

    #Crea el árbol
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]

        #Selecciona el  atributo que mejor particiona el dataset (con ayuda de la función info_gain)
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        #Estructura del árbolCreate the tree structure.
        #La raíz corresponde al atributo con mayor ganancia de información (menor entropía)
        tree = {best_feature:{}}

        #Actualiza los atributos quitanto el best_feature
        features = [i for i in features if i != best_feature]

        #Crea una rama abajo del nodo para cada valor del atributo raíz
        for value in np.unique(data[best_feature]):
            value = value
            #Particiona el dataset por el valor que genera el atributo con mayor ganancia de info, y crea subconjuntos
            sub_data = data.where(data[best_feature] == value).dropna()

            #Llama recursivamente al ID3 con cada uno de esos subconjuntos
            subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_class)

            #Agrega el subárbol al árbol
            tree[best_feature][value] = subtree

        return (tree)

#############################################################
########################## PREDICT ##########################
#############################################################

# Predice información nueva
#
# Entrada:
#        query = Recibe la información nueva en forma de diccionario {"atributo":valor,...}
#        tree = El árbol generado por id3
#
# Salida:
#        1. Una lista de sus predicciones

def predict(query,tree,default):
    #Checa para cada atributo en el query si existe en el árbol, si no existe regresa el valor default
    for key in list(query.keys()):
        if key in list(tree.keys()):
            if not(query[key] in tree[key]) and key == 'absences':
                if query[key]<2: query[key] = 0
                elif query[key]<4: query[key] = 2
                elif query[key]<6: query[key] = 4
                elif query[key]<8: query[key] = 6
                elif query[key]<10: query[key] = 8
                elif query[key]<12: query[key] = 10
                elif query[key]<14: query[key] = 12
                elif query[key]<25: query[key] = 14
                elif query[key]<54: query[key] = 25
                else: query[key] = 54
            elif not(query[key] in tree[key]) and key == 'G1':
                if query[key]==18: query[key] = 17
                elif query[key]==20: query[key] = 19
                elif query[key] < 5: query[key] = 5
            elif not(query[key] in tree[key]) and key == 'G2':
                if query[key]==17: query[key] = 18
                elif query[key]==20: query[key] = 19
                elif query[key] < 5: query[key] = 5
            elif not(query[key] in tree[key]) and key == 'age':
                if query[key] > 18: query[key] = 16
            #Hay que decirle a la predicción que hacer en caso de encontrar un atributo con un valor
            #no registrado en el arbol. En este caso asignamos el valor de default
            try:
                result = tree[key][query[key]]
            except:
                return default

            #Recorre el árbol comparando el query con los valores del árbol
            result = tree[key][query[key]]
            #Llamamos recursivamente a la función predict con el nuevo subárbol
            if isinstance(result,dict):
                return predict(query,result,default)

            else:
                return result

#############################################################
########################### TEST ############################
#############################################################

# Crea una nueva consulta convirtiendo las columnas de atributo a un diccionario
#
# Entrada:
#        data = dataset
#        tree = árbol previamente construido
#        target_attribute_name = nombre de
#
# Salida:
#        1. dataframe con las predicciónes

def testB(data,tree,target_attribute_name):
    queries = data.iloc[:,:-1].to_dict(orient = "records")

    #Crear un dataframe vacío en cuyas columnas se almacena la predicción
    predicted = pd.DataFrame(columns=["predicted"])

    #Calcula el accuracy de la predicción
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,'pass')

    return predicted

def testE(data,tree,target_attribute_name):
    queries = data.iloc[:,:-1].to_dict(orient = "records")

    #Crear un dataframe vacío en cuyas columnas se almacena la predicción
    predicted = pd.DataFrame(columns=["predicted"])

    #Calcula el accuracy de la predicción
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,'sufficient')

    return predicted

#############################################################
########################            #########################
######################                #######################
#####################  -PYGAME CONFIG  ######################
######################                #######################
########################            #########################
#############################################################

x_width, y_width = 1024, 720
pygame.init()
screen = pygame.display.set_mode((x_width, y_width))
pygame.display.set_caption("ID3")
# pygame.display.set_icon(icono)
font_name = assets_path+'Fantasque-Regular.ttf'
FontT=pygame.font.Font(font_name, 36)
FontM=pygame.font.Font(font_name, 20)
FontS=pygame.font.Font(font_name, 16)
FontXS=pygame.font.Font(font_name, 14)
FontXXS=pygame.font.Font(font_name, 12)
datasets = ['Binary', 'Binary G1', 'Binary G2', 'Erasmus', 'Erasmus G1', 'Erasmus G2']

widths =  [[188, 14], [240, 12], [240, 12], [325, 12], [492,12], [325, 12]]
colors = [[],[],[]]
for i in range(21):
    colors[0].append(rand.randint(0, 8)*10)
    colors[1].append(rand.randint(0, 8)*8)
    colors[2].append(rand.randint(0, 8)*10)

features = {
  ##########################################################################################################
  'Género': 'sex',                               # sex:         F,M
  'Edad': 'age',                                 # age:         15:22
  'Lugar de residencia': 'address',              # address:     R,U
  'Familiares en casa' : 'famsize',              # address:     LE3, GT3
  'Estado de los padres' : 'Pstatus',            # Pstatus:     A, T
  ##########################################################################################################
  'Educación de la madre' : 'Medu',              # Medu:        4th grade, 5th - 9th grade, higher, secondary
  'Educación del padre' : 'Fedu',                # Fedu:        4th grade, 5th - 9th grade, higher, secondary
  'Trabajo de la madre' : 'Mjob',                # Mjob:        at_home, health, services, teacher, other
  'Trabajo del padre' : 'Fjob',                  # Fjob:        at_home, health, services, teacher, other
  ##########################################################################################################
  'Razón de estudiar' : 'reason',                # reason:      course, reputation, home, other
  'Tiempo de estudio' : 'studytime',             # studytime:   < 2 hrs, 2 to 5 hrs, 5 to 10 hrs, > 10 hrs
  'Beca' : 'schoolsup',                          # schoolsup:   yes, no
  'Apoyo familiar' : 'famsup',                   # famsup:      yes, no
  ##########################################################################################################
  'Sale con frecuencia' : 'goout',               # goout:       very low, low, medium, high, very high
  'Consumo de alcohol entre semana' : 'Dalc',    # Dalc:        very low, low, medium, high, very high
  'Consumo de alcohol en fin semana' : 'Walc',   # Walc:        very low, low, medium, high, very high
  ##########################################################################################################
  'Relación con la familia' : 'famrel',          # famrel:      very bad, bad, good, very good, excellent
  'Salud general' : 'health',                    # health:      very bad, bad, good, very good, excellent
  ##########################################################################################################
  'Número de faltas' : 'absences',               # absences:    0:54
  'Calificación del 1º parcial' : 'G1',          # G1:          0:20
  'Calificación del 2º parcial' : 'G2'}          # G2:          0:20

questions = [['Género', 'Edad', 'Lugar de residencia', 'Familiares en casa', 'Estado de los padres'],
             ['Educación de la madre', 'Educación del padre', 'Trabajo de la madre', 'Trabajo del padre'],
             ['Razón de estudiar', 'Tiempo de estudio', 'Beca', 'Apoyo familiar'],
             ['Sale con frecuencia', 'Consumo de alcohol entre semana', 'Consumo de alcohol en fin semana'],
             ['Relación con la familia', 'Salud general'],
             ['Número de faltas', 'Calificación del 1º parcial', 'Calificación del 2º parcial']]

q_choises = [[['Hombre','Mujer'],[15,16,17,18,19,20,21,22],['Rural','Urbano'],['Menos de, o 3','Más de 3'],['Separados','Juntos']],
             [['Prim', 'Secu', 'Prep', 'Sup'],['Prim', 'Secu', 'Prep', 'Sup'],['Salud','Gob','Edu','Casa','Otro'],['Salud','Gob','Edu','Casa','Otro']],
             [['Intelecto','Renombre','Familia','Otro'],['<2 hrs','2-5 hrs','5-10 hrs', '>10 hrs'],['Si','No'],['Si','No']],
             [['Muy baja', 'Baja', 'Media','Alta','Muy alta'], ['Muy baja', 'Baja', 'Media','Alta','Muy alta'], ['Muy baja', 'Baja', 'Media','Alta','Muy alta']],
             [['Muy mala', 'Mala', 'Neutral','Buena','Muy buena'], ['Muy mala', 'Mala', 'Neutral','Buena','Muy buena']],
             [[0,2,4,6,8,10,12,14,16,18,20,25,50],[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]]

q_ans = {
    'Hombre':'H', 'Mujer':'M', '15':15, '16': 16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'Rural':'R', 'Urbano':'U',
    'Menos de, o 3':'LE3', 'Más de 3':'GT3', 'Separados':'A', 'Juntos':'T', 'Prim':'4th grade', 'Secu':'5th - 9th grade', 'Prep':'secondary',
    'Sup':'higher', 'Salud':'health', 'Gob':'services', 'Edu':'teacher','Casa':'at_home','Otro':'other', 'Intelecto':'course',
    'Renombre':'reputation', 'Familia':'home', 'Otro':'other', '<2 hrs':'< 2 hrs', '2-5 hrs':'2 to 5 hrs', '5-10 hrs':'5 to 10 hrs',
    '>10 hrs':'> 10 hrs', 'Si':'yes', 'No':'no', 'Muy baja':'very low', 'Baja':'low', 'Media':'medium', 'Alta':'high', 'Muy alta':'very high',
    'Muy mala':'very bad', 'Mala':'bad', 'Neutral':'good', 'Buena':'very good','Muy buena':'excellent', '0':0, '2':2, '4':4, '5':5, '6':6,
    '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, '14':14, '25':25, '50':50}

tree_bin, tree_bin_g1, tree_bin_g2, tree_era, tree_era_g1, tree_era_g2, target_b, target_e, binp = '','','','','','','','',''

ans_data = ['H',15,'R','LE3','A','4th grade','4th grade','health','health','course','< 2 hrs','yes','yes','very low',
            'very low','very low','very bad','very bad',0,5,5]
ans_no = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#############################################################
###################### PYGAME SCREENS #######################
#############################################################

def Title(word):
    letter=FontT.render(word, False, (40,40,40),(215, 215, 215))
    w, h = FontT.size(word)
    screen.blit(letter, (x_width/2-w/2,h/2-10))

def Loading(word):
    word = 'Loading '+ word +'...'
    screen.fill((215, 215, 215))
    letter=FontT.render(word, False, (40,40,40),(215, 215, 215))
    w, h = FontT.size(word)
    screen.blit(letter, (x_width/2-w/2,y_width/2-h/2-36))
    pygame.display.update()

def UnderButtons(click, mouse_x, mouse_y):
    global scroll_y
    button = []
    bw, mw = 152, 16
    for i in range(len(datasets)):
        button.append(pygame.Rect((mw+(bw+mw)*i, 674, bw, 32)))
        letter=FontM.render(datasets[i], False, (215, 215, 215), (40,40,40))
        w, h = FontM.size(datasets[i])
        pygame.draw.rect(screen, (40, 40, 40), button[i], 0, 2)
        screen.blit(letter, (bw/2+mw+(bw+mw)*i-w/2,680))
    for i in range(len(button)):
        if button[i].collidepoint(mouse_x, mouse_y):
            letter=FontM.render(datasets[i], False, (215, 215, 215), (70,40,70))
            w, h = FontM.size(datasets[i])
            pygame.draw.rect(screen, (70,40,70), button[i], 0, 2)
            screen.blit(letter, (bw/2+mw+(bw+mw)*i-w/2,680))
            if(click):
                trees = [tree_bin, tree_bin_g1, tree_bin_g2, tree_era, tree_era_g1, tree_era_g2]
                ShowTree(datasets[i], trees[i])
                scroll_y = 0

def Back(click, mouse_x, mouse_y):
    button = pygame.Rect(12,12,100,20)
    letter=FontS.render('Regresar', False, (215, 215, 215), (40,40,40))
    w, h = FontS.size('Regresar')
    pygame.draw.rect(screen, (40,40,40), button)
    screen.blit(letter, (62-w/2,16))
    if button.collidepoint(mouse_x, mouse_y):
        letter=FontS.render('Regresar', False, (215, 215, 215), (70,40,70))
        w, h = FontS.size('Regresar')
        pygame.draw.rect(screen, (70,40,70), button)
        screen.blit(letter, (62-w/2,16))
        if(click):
            return True
    return False


#############################################################
########################### INIT ############################
#############################################################

def Start():
    global tree_bin, tree_bin_g1, tree_bin_g2, tree_era, tree_era_g1, tree_era_g2, target_b, target_e, binp
    training_data_binary = train_test_split(data_binary)[0]
    training_data_erasmus = train_test_split(data_erasmus)[0]

    header = data_binary.columns.tolist()
    target_b = header[32]
    target_e = data_erasmus.columns.tolist()[32]

    Loading(datasets[0])
    tree_bin = ID3(training_data_binary, training_data_binary, header[0:30], target_b)
    Loading(datasets[1])
    tree_bin_g1 = ID3(training_data_binary, training_data_binary, header[0:31], target_b)
    Loading(datasets[2])
    tree_bin_g2 = ID3(training_data_binary, training_data_binary, header[0:32], target_b)
    Loading(datasets[3])
    tree_era = ID3(training_data_erasmus, training_data_erasmus, header[0:30], target_e)
    Loading(datasets[4])
    tree_era_g1 = ID3(training_data_erasmus, training_data_erasmus, header[0:31], target_e)
    Loading(datasets[5])
    tree_era_g2 = ID3(training_data_erasmus, training_data_erasmus, header[0:32], target_e)

    binp = training_data_binary.head(1)
    Main()

#############################################################
########################### START ###########################
#############################################################

def Main():
  click  = False
  while True:
    screen.fill((215, 215, 215))
    mouse_x, mouse_y = pygame.mouse.get_pos()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        exit()
      if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1: click = True

    Title("Desempeño de estudiantes")
    SelectionButtons(click, mouse_x, mouse_y)
    UnderButtons(click, mouse_x, mouse_y)

    click = False
    pygame.display.update()

#############################################################
###################### SELECTION GRID #######################
#############################################################

def SelectionButtons(click, mouse_x, mouse_y):
    Labels()
    Options(click, mouse_x, mouse_y)
    Calculate(click, mouse_x, mouse_y)

def Labels():
    ph = 55
    color_p = 0
    for i in range(len(questions)):
        for j in range(len(questions[i])):
            letter=FontS.render(questions[i][j], False, (215, 215, 215), (40+colors[0][color_p], 40+colors[1][color_p], 40+colors[2][color_p]))
            w, h = FontS.size(questions[i][j])
            pygame.draw.rect(screen, (40+colors[0][color_p], 40+colors[1][color_p], 40+colors[2][color_p]), (widths[i][1]+(widths[i][0]+widths[i][1])*j, ph+(i*95), widths[i][0], 32), 0, 2)
            screen.blit(letter, (widths[i][0]/2+widths[i][1]+(widths[i][0]+widths[i][1])*j-w/2,ph+(i*95)+8))
            color_p += 1

def Options(click, mouse_x, mouse_y):
    global ans_data, ans_no
    ph = 101
    choises = []
    opt = -1
    for i in range(len(q_choises)):
        choises.append([])
        for j in range(len(q_choises[i])):
            choises[i].append([])
            pygame.draw.rect(screen, (195, 195, 195), (widths[i][1]+(widths[i][0]+widths[i][1])*j, ph+(i*95), widths[i][0], 32))
            opt += 1
            for k in range(len(q_choises[i][j])):
                if ans_no[opt] == k: set_color = (130, 130, 195)
                else: set_color = (195, 195, 195)
                choises[i][j].append(pygame.Rect(widths[i][1]+(widths[i][0]+widths[i][1])*j+(widths[i][0]/len(q_choises[i][j]))*k, ph+(i*95), widths[i][0]/len(q_choises[i][j]), 32))
                w, h = FontXS.size(str(q_choises[i][j][k]))
                if w+8 > widths[i][0] / len(q_choises[i][j]):
                    letter=FontXXS.render(str(q_choises[i][j][k]), False, (40,40,40), set_color)
                    w, h = FontXXS.size(str(q_choises[i][j][k]))
                    ph+=2
                else: letter=FontXS.render(str(q_choises[i][j][k]), False, (40,40,40), set_color)
                pygame.draw.rect(screen, set_color, choises[i][j][k])
                screen.blit(letter, (widths[i][0]/(2*len(q_choises[i][j]))+widths[i][1]+(widths[i][0]+widths[i][1])*j+widths[i][0]/len(q_choises[i][j])*k-w/2,ph+(i*95)+8))
                ph=101

    opt = -1
    for i in range(len(choises)):
        for j in range(len(choises[i])):
            opt += 1
            for k in range(len(choises[i][j])):
                if choises[i][j][k].collidepoint(mouse_x, mouse_y):
                    w, h = FontXS.size(str(q_choises[i][j][k]))
                    if w+8 > widths[i][0] / len(q_choises[i][j]):
                        letter=FontXXS.render(str(q_choises[i][j][k]), False, (40,40,40), (175, 155, 175))
                        w, h = FontXXS.size(str(q_choises[i][j][k]))
                        ph+=2
                    else: letter=FontXS.render(str(q_choises[i][j][k]), False, (40,40,40), (175, 155, 175))
                    pygame.draw.rect(screen, (175, 155, 175), choises[i][j][k])
                    screen.blit(letter, (widths[i][0]/(2*len(q_choises[i][j]))+widths[i][1]+(widths[i][0]+widths[i][1])*j+widths[i][0]/len(q_choises[i][j])*k-w/2,ph+(i*95)+8))
                    if(click):
                        ans_no[opt] = k
                        ans_data[opt] = q_ans[str(q_choises[i][j][k])]
                    ph=101

def Calculate(click, mouse_x, mouse_y):
    button = pygame.Rect(256,625,512,32)
    letter=FontM.render('Calcular', False, (215, 215, 215), (40,40,40))
    w, h = FontM.size('Calcular')
    pygame.draw.rect(screen, (40,40,40), button, 0, 2)
    screen.blit(letter, (512-w/2,630))
    if button.collidepoint(mouse_x, mouse_y):
        letter=FontM.render('Calcular', False, (215, 215, 215), (70,40,70))
        w, h = FontM.size('Calcular')
        pygame.draw.rect(screen, (70,40,70), button, 0, 2)
        screen.blit(letter, (512-w/2,630))
        if(click):
            d = 0
            for i in questions:
                for j in i:
                    binp.at[0, features[j]] = ans_data[d]
                    d += 1
            erap = binp.rename(columns={"classification": "erasmus"})
            pdb = testB(binp, tree_bin, target_b)
            pdb1 = testB(binp, tree_bin_g1, target_b)
            pdb2 = testB(binp, tree_bin_g2, target_b)
            pde = testE(erap, tree_era, target_e)
            pde1 = testE(erap, tree_era_g1, target_e)
            pde2 = testE(erap, tree_era_g2, target_e)

            Results(pdb.at[0, 'predicted'], pdb1.at[0, 'predicted'], pdb2.at[0, 'predicted'], pde.at[0, 'predicted'], pde1.at[0, 'predicted'], pde2.at[0, 'predicted'])

#############################################################
########################## RESULTS ##########################
#############################################################

def ShowTree(title, tree):
      click, scu, scd  = False, False, False
      while True:
        screen.fill((215, 215, 215))
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            pygame.quit()
            exit()
          if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: click = True
            elif event.button == 4: scu = True
            elif event.button == 5: scd = True

        Title(title)
        if Back(click, mouse_x, mouse_y): break
        TreeView(scu, scd, tree)

        click, scu, scd = False, False, False
        pygame.display.update()

scroll_y = 0

def TreeView(scu, scd, tree):
    global scroll_y
    out = pp.pformat(tree).split('\n')
    w, h = FontS.size(out[0])
    fh = max((h+10)*len(out)+10,600)
    options = pygame.surface.Surface((976, fh))
    options.fill((230, 230, 230))
    for i in range(len(out)):
        letter=FontXS.render(out[i], False, (40,40,40),(230, 230, 230))
        options.blit(letter, (8,10 + i*(h+10)))

    if(scu): scroll_y = max(scroll_y - 20, 0)
    elif(scd): scroll_y = min(scroll_y + 20, fh-600)

    pygame.draw.rect(screen, (40, 40, 40), (23, 95, 978, 602))
    screen.blit(options, (24, 96), (0,scroll_y,976,600))


#############################################################
########################## RESULTS ##########################
#############################################################

def Results(b,b1,b2,e,e1,e2):
      click  = False
      while True:
        screen.fill((215, 215, 215))
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            pygame.quit()
            exit()
          if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: click = True

        Title("Predicción")
        if Back(click, mouse_x, mouse_y): break
        ResLabels()
        ####################### 2 #######################
        l1 = FontM.render(b2, False, (40,40,40), (215, 215, 215))
        w1, h1 = FontM.size(b2)
        screen.blit(l1, (263.5-w1/2,200))
        l2 = FontM.render(e2, False, (40,40,40), (215, 215, 215))
        w2, h2 = FontM.size(e2)
        screen.blit(l2, (760.5-w2/2,200))
        ####################### 1 #######################
        l1 = FontM.render(b1, False, (40,40,40), (215, 215, 215))
        w1, h1 = FontM.size(b1)
        screen.blit(l1, (263.5-w1/2,200+150))
        l2 = FontM.render(e1, False, (40,40,40), (215, 215, 215))
        w2, h2 = FontM.size(e1)
        screen.blit(l2, (760.5-w2/2,200+150))
        ###################### 0 #######################
        l1 = FontM.render(b, False, (40,40,40), (215, 215, 215))
        w1, h1 = FontM.size(b)
        screen.blit(l1, (263.5-w1/2,200+300))
        l2 = FontM.render(e, False, (40,40,40), (215, 215, 215))
        w2, h2 = FontM.size(e)
        screen.blit(l2, (760.5-w2/2,200+300))

        click = False
        pygame.display.update()

def ResLabels():
    ph = 55
    color_p = 0
    for i in range(3):
        letterA=FontM.render(datasets[2-i], False, (215, 215, 215), (40+colors[0][color_p], 40+colors[1][color_p], 40+colors[2][color_p]))
        letterB=FontM.render(datasets[5-i], False, (215, 215, 215), (40+colors[0][color_p+1], 40+colors[1][color_p+1], 40+colors[2][color_p+1]))
        wa, ha = FontM.size(datasets[2-i])
        wb, hb = FontM.size(datasets[5-i])
        pygame.draw.rect(screen, (40+colors[0][color_p], 40+colors[1][color_p], 40+colors[2][color_p]), (30,150+150*i,467,32))
        pygame.draw.rect(screen, (40+colors[0][color_p+1], 40+colors[1][color_p+1], 40+colors[2][color_p+1]), (527,150+150*i,467,32))
        screen.blit(letterA, (263.5-wa/2,150+150*i))
        screen.blit(letterB, (760.5-wb/2,150+150*i))
        color_p += 3

Start()
