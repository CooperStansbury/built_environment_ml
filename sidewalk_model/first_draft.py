

# %%
import pandas as pd
import numpy as np
import json


from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# eval
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support
# %%

FILEPATH = '../washtenaw_tmp/washtenaw_sidewalks.json'

def load_annotations(filepath):
    """A function to get annotations from json object.
    
    NOTE:
        - "Download file would be a text file where each line is a 
           JSON containing the image URL and the classes marked for 
           the image."
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    new_rows = []

    for annotated_file in open(filepath):
        file_dict = json.loads(annotated_file)

        annotated_filename = file_dict['content'].split("___")[1]
        tile_id = annotated_filename.replace('tile_', '').split(".")[0]

        if not file_dict['annotation'] is None:
            annotations = file_dict['annotation']['labels']
        else:
            annotations = ['None']

        row = {
            'tile_id':int(tile_id),
            'file_name':annotated_filename,
            'annotations':annotations
        }

        new_rows.append(row)
    
    df = pd.DataFrame(new_rows)
    mlb = MultiLabelBinarizer()
    one_hots = pd.DataFrame(mlb.fit_transform(df['annotations']),columns=mlb.classes_)

    df = pd.concat([df, one_hots], axis=1)

    return df
    


df = load_annotations(FILEPATH)
# reorder
df = df.sort_values(by='tile_id').reset_index(drop=True)
df.head()

# %%

df['sidewalk'].value_counts(normalize=True)


# %%

X = np.load('../washtenaw_tmp/tiles.npy')
X = np.moveaxis(X, 1, -1)

# %%
print(df.shape)
print(X.shape)

# %%

X_train, X_test, train,test = train_test_split(X, df, test_size=0.33, random_state=42)

y_train = train['sidewalk']
y_test = test['sidewalk']

print(f"X_train.shape {X_train.shape}")
print(f"X_test.shape {X_test.shape}")
print(f"y_train.shape {y_train.shape}")
print(f"y_test.shape {y_test.shape}")
print()

X_train = np.asarray(list(X_train)) 
X_test = np.asarray(list(X_test))
y_train = np.asarray(list(y_train))
y_test = np.asarray(list(y_test))

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# create val data
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=0.33, 
                                                  random_state=42)


print(f"X_train.shape {X_train.shape}")
print(f"X_test.shape {X_test.shape}")
print(f"X_val.shape {X_val.shape}")
print(f"y_val.shape {y_val.shape}")
print(f"y_train.shape {y_train.shape}")
print(f"y_test.shape {y_test.shape}")

# %%

def get_classification_metrics(model, X_test, y_test):
    """A function to 'pprint' classification metrics (binary)"""
    y_proba = model.predict(X_test)[:,1]
    y_test = y_test[:, 1]

    # compute tpr/fpr at every thresh
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # get optimal threshold by AUCROC
    optimal_idx = np.argmax(tpr - fpr)

    optimal_threshold = thresholds[optimal_idx]
    aucroc = roc_auc_score(y_test, y_proba)

    # compute predictions based on optimal threshold
    y_pred = np.where(y_proba >= optimal_threshold, 1, 0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # get precision/recall
    rate = y_test.mean()
    precision = tp / (tp + fn * (1 / rate - 1))
    recall = tp / (tp + fn * (1 / rate - 1))
    f1 = 2 * tp / (2*tp + fp + fn)

    res_dict = {
        'optimal_threshold':optimal_threshold,
        'true negatives': tn,
        'true positives': tp,
        'false positives': fp,
        'false negatives': fn,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'F1-score' : f1,
        'precision': precision,
        'recall': recall,
        'AUCROC' : aucroc,
    }

    res = pd.DataFrame.from_dict(res_dict, orient='index')
    return res

# %%

BATCH_SIZE = 100

input_shape = X_train[0].shape
print(input_shape)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dropout(0.2))    

# model.add(Conv2D(32, (17, 17), 
#                  padding='same', 
#                  activation='relu'))

model.add(Conv2D(32, (11, 11), 
                 padding='same', 
                 activation='relu'))

# model.add(Conv2D(10, (5, 5), 
#                  padding='same', 
#                  activation='relu'))

model.add(MaxPooling2D(pool_size=8))  
model.add(Dropout(0.2))  
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['AUC'])
            #   metrics=['Precision'])

model.summary()

mod2 = model.fit(X_train, 
                 y_train, 
                 validation_data=(X_val,y_val), 
                 steps_per_epoch= X_train.shape[0] // BATCH_SIZE,
                 epochs=2)

res = get_classification_metrics(mod2.model, X_test, y_test)
res

# %%


y_proba = model.predict(X_test)[:,1]
# %%

DECISION = 0.5
N_SAMPLE = 10

positive_mask = (y_proba > DECISION)
tmp = test[positive_mask].sample(N_SAMPLE)
tmp[['tile_id', 'sidewalk']].sort_values(by='tile_id')

# %%
