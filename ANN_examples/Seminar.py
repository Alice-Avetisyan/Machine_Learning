# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
import matplotlib.pyplot as plt

# **Տվյալների ընտրություն**
# Այսետղ ներբեռնում ենք fashion_mnist (Modified National Institute of Standards and Technology database) տվյալները,
# և բաժանում ենք դրանք ուսուցման և տեստավորման համար։

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Տվյալների նորմալիզացիա
# mnist-ը կազմված է սև և սպիտակ նկարներից, որտեղ փիկսելների արժեքները 0-ից 255 դիսկրետ արժեքներ են։
# Նորմալիզացիա անելուց հետո այդ միջակայքը վերածվում է 0-ից 1-ի։

X_train, X_test = X_train / 255.0, X_test / 255.0

print(X_train.shape)  # output: (60000, 28, 28)

# **Տվյալների մշակում**
# Ի սկզբանե, հատկանիշների (feature, X) ֆորմատը (60000, 28, 28) է՝ նմուշների քանկը 60000, իսկ տողերի և սյուների քանակը 28։
# Մոդելը ստեղծելու համար մեզ հարկավոր է ևս մեկ արժեք, որը կնիշի տվյալների ալիքը։
# Որպեսզի տվյալները վերափոխենք, ես օգտագործել եմ reshape մեթոդը։

X_train = X_train.reshape(len(X_train), 28, 28, 1)
X_test = X_test.reshape(len(X_test), 28, 28, 1)

# Իսկ թիրախ/պիտակների (label, y) ֆորամտավորումը կախված է մոդելի loss ատրիբուտից,
# և քանի որ օգտագործվել է այնպիսի ատրիբուտ, որը ակնկալում է դիսկրետ թիվ,
# ապա հարկավոր է մեր թիրախ տվյալների արժեքները կոնվերտացնել 0 և 1։

y_train = tf.keras.utils.to_categorical(y_train, 10) # one-hot format
y_test = tf.keras.utils.to_categorical(y_test) # one-hot format

# Ավելացնում ենք մեզ հարկավոր գրադարանները՝

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D
from tensorflow.keras import Sequential


# Ստեղծում ենք լիստ, որը պարունակում է դասերի անվանումները`

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# **Տվյալների վիզուալացում**
# Մեր տվյալները մշակելուց հետո կարող ենք օգտագործել վիզուալացման matplotlib գրադարանը։

# output: image belonging to the 0 index
plt.figure()
plt.imshow(X_train[0][..., 0])  # imshow expects 2D or rgb vector
plt.colorbar()
plt.grid(False)
plt.show()

# output: first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i][..., 0], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(y_train[i])])
plt.show()


# **Մոդելի մշակում**
# Որպեսզի կարողանանք կատարենք մեր տվյալների դասակարգել, նախ և առաջ հարկավոր է ստեղծել մոդել։
# Այս նախագծում օգտագործվել է կոնվոլուցիոն նեյրոնային ցանցը, քանի որ այն օպտիմալ ընտրությունն է պատկերների դասակարգելու համար։
# Ստեղծված կոնվոլուցիոն մոդելը ունի երեք շերտ և ուսուցման մակարդակի ատրիբուտ (lr), որի օգտագործումը մնեք կտեսնենք միքիչ հետո։

def make_model(lr=0.001):
    model = Sequential([Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
                       MaxPool2D(2),
                       Conv2D(32, 3, padding='same', activation='relu'),
                       MaxPool2D(2),
                       tf.keras.layers.Flatten(),
                       Dense(10, 'softmax')])
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# **Մոդելի ուսուցում**
# Մոդել ստեղծելուց հետո սկսում ենք նրա ուսուցումը օգտագործելով 32, 128, 256, 1024, 4096 և 1 խմբաքանակները,
# և վերջում համեմատում ենք ուցուման կատարած աշխատանքը։

cnn_model = make_model()

from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger('model_32.csv')
history_32 = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[csv_logger]) 

cnn_model = make_model()  # reinstantiating the model
csv_logger = CSVLogger('model_128.csv')
history_128 = cnn_model.fit(X_train, y_train, epochs=10, batch_size=128, callbacks=[csv_logger])


cnn_model = make_model()  # reinstantiating the model
csv_logger = CSVLogger('model_256.csv')
history_256 = cnn_model.fit(X_train, y_train, epochs=10, batch_size=256, callbacks=[csv_logger])


cnn_model = make_model()  # reinstantiating the model
csv_logger = CSVLogger('model_1024.csv')
history_1024 = cnn_model.fit(X_train, y_train, epochs=10, batch_size=1024, callbacks=[csv_logger])


cnn_model = make_model()
csv_logger = CSVLogger('model_4096.csv')
history_4096 = cnn_model.fit(X_train, y_train, epochs=10,batch_size=4096, callbacks=[csv_logger])


cnn_model = make_model()
csv_logger = CSVLogger('model_full_batch.csv')
history_full_batch = cnn_model.fit(X_train, y_train, epochs=10, batch_size=60000, callbacks=[csv_logger])


cnn_model = make_model()
csv_logger = CSVLogger('model_1.csv')
history_1 = cnn_model.fit(X_train, y_train, epochs=10, batch_size=1, callbacks=[csv_logger])


# **Մոդելի վիզուալացում**
# Մոդելների ուսուցումից հետո և ներքևում պատկերվածին նայելով կարող ենք ենթադրություն անել,
# որ ստեղծվավծ մոդելներից ամենա օպտիմալ loss-ի արդյունք ունի batch_size=32 ունեցող մոդելը,
# քանի որ մոդելի ուսուցման ժամանակ կշիռների տատանումը ավելի քիչ է, համեմատած batch_size=1 խմբաքանակի հետ։
# Իսկ մեծ խմբաքանակներում դիտվում են քիչ կորուստի փոփոխություններ։


plt.figure(figsize=(8, 8))
#  32, 128, 256, 1024, 4096, 60000, 1
plt.plot(list(range(10)), history_1.history['loss'], label='Training Loss 1')

plt.plot(list(range(10)), history_32.history['loss'], label='Training Loss 32')

plt.plot(list(range(10)), history_128.history['loss'], label='Training Loss 128')

plt.plot(list(range(10)), history_256.history['loss'], label='Training Loss 256')

plt.plot(list(range(10)), history_1024.history['loss'], label='Training Loss 1024')

plt.plot(list(range(10)), history_4096.history['loss'], label='Training Loss 4096')

plt.plot(list(range(10)), history_full_batch.history['loss'], label='Training Loss 60000')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()


# Կատարենք ևս մի քանի փորձարկումներ։ Այս անգամ օգտագործենք նաև ուսուցման մակարդակը և նվազացնենք այն։

cnn_model = make_model(0.0001)
csv_logger = CSVLogger('model_1_low_lr.csv')
history_1_low_lr = cnn_model.fit(X_train, y_train, epochs=10, batch_size=1, callbacks=[csv_logger])


# Այս մոդելի ուսուցման ժամանակ կուրուստի մակարդակը նվազել է շատ ավելին քան մեկ խմբաքանակով
# (default learning rate 0.001) մոդելից։
# Սակայն, իր մակարդակը չի անցել 32-ին, որի ուսուցման մակարդակը նույնպես 0.001 է։


plt.figure(figsize=(8, 8))

plt.plot(list(range(10)), history_1.history['loss'], label='Training Loss 1')

plt.plot(list(range(10)), history_32.history['loss'], label='Training Loss 32')

plt.plot(list(range(10)), history_1_low_lr.history['loss'], label='Training Loss 1 w low lr')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()


# Կրկին փոքրացնենք ուսուցման մակարդակը օգտագործելով մեկ քանակի խմբաքանակը։ 


cnn_model = make_model(0.00005)
csv_logger = CSVLogger('model_1_extra_low_lr.csv')
history_1_extra_low_lr = cnn_model.fit(X_train, y_train, epochs=10, batch_size=1, callbacks=[csv_logger])


# Կրկին այս մոդելի կորուստի մակարդակը չի հասնում 32 խմբաքանակով ուսուցում կատարված մոդելին։


plt.figure(figsize=(8, 8))

plt.plot(list(range(10)), history_1.history['loss'], label='Training Loss 1')

plt.plot(list(range(10)), history_32.history['loss'], label='Training Loss 32')

plt.plot(list(range(10)), history_1_low_lr.history['loss'], label='Training Loss 1 w low lr')

plt.plot(list(range(10)), history_1_extra_low_lr.history['loss'], label='Training Loss 1 w extra low lr')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()


# Այստեղ մենք տենում ենք, որ ուսուցման մակարդակը ունի ազդեցություն ուսուցման վրա, և խմաբաքանակը նույնպես ունի որոշակի
# ազդեցություններ, բայց ի՞նչպես են այս երկու ատրիբուտները իրար հետ "համգործակցում"։


def decay_lr(epoch):
    initial_lr = 0.001
    k = 0.1 
    lr = initial_lr * np.exp(-k * epoch)
    
    return lr


# Այս մեթոդը իրենից ներկայացնում է ուսուցման մակարդակի նվազեցում, որը վերադարձնում է փոփոխված՝ նվազեցված ուսուցման մակարդակը։
# Նվազեցված ուսուցման մակարդակը ուսուցաման ամեն իտեռացիայի ժամանակ գնալով կշիռների վրա կկատարի ավելի ու ավելի քիչ փոփոխություններ։ Եվս մեկ նոր մոդել ստեղծելիս, մենք կիրառում ենք այս մեթոդը ամեն իտեռացիաի ժամանակ։


cnn_model = make_model()
csv_logger = CSVLogger('model_1_lr_decay.csv')
lr_scheduler= tf.keras.callbacks.LearningRateScheduler(decay_lr, 1)
history_1_lr_decay = cnn_model.fit(X_train, y_train, epochs=10, batch_size=1, callbacks=[csv_logger, lr_scheduler])


# Ապա այստեղ նկատվում է կորուստի մեծ փոփխություններ։ Քանի որ ուսուցման մակարդակը կարավարում է կշիռների թարմացումը։
# Որքան ցածր է արժեքը, այնքան դանդաղ է փոփոխություն կատարվում։


plt.figure(figsize=(8, 8))

plt.plot(list(range(10)), history_1.history['loss'], label='Training Loss 1')

plt.plot(list(range(10)), history_32.history['loss'], label='Training Loss 32')

plt.plot(list(range(10)), history_1_low_lr.history['loss'], label='Training Loss 1 w low lr')

plt.plot(list(range(10)), history_1_extra_low_lr.history['loss'], label='Training Loss 1 w extra low lr')

plt.plot(list(range(10)), history_1_lr_decay.history['loss'], label='Training Loss 1 w decay')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()


cnn_model = make_model(0.01)
csv_logger = CSVLogger('model_4096_lr_0,01.csv')
history_4096_lr_001 = cnn_model.fit(X_train, y_train, epochs=10, batch_size=4096, callbacks=[csv_logger])


# Եվ վերջապես, այս պատեկում մենք դիտակում ենք 4096 խմբաքանակի տարբերությունը սկզբնական ուսուցման մակարդակի և 0.01 մակարդակի հետ։
# Համաձայն փորձին մենք կարող ենք բարձր խմբաքանակ ունեցող մոդելի մոտ նաև ավելացնել իր ուսուցման մակարդակը։
# Բայց ուսուցման մակարդկաի ավելացումը ունի սահմանափակումներ, քանի որ որոշակի արժեքից հետո կուրուստի մակարդկարը կանգնում է ու դադարում է նվազել։


plt.figure(figsize=(8, 8))

plt.plot(list(range(10)), history_4096.history['loss'], label='Training Loss 4096')

plt.plot(list(range(10)), history_4096_lr_001.history['loss'], label='Training Loss 4096 w 0.01 lr')

plt.plot(list(range(10)), history_1024.history['loss'], label='Training Loss 1024')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

