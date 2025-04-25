# -*- coding: utf-8 -*-
"""

#Classificazione del Rischio di Malattie Cardiache

## Introduzione e Scopo del Progetto

### Descrizione del Dataset
Per questo progetto ho utilizzato un dataset pubblico: "Health and Lifestyle" (https://www.kaggle.com/datasets/mahdimashayekhi/health-and-lifestyle-dataset).

Contiene informazioni sulla salute e sullo stile di vita di diverse persone. Tra le features abbiamo: *età, genere, altezza, peso, BMI, passi giornalieri, ore di esercizio settimanali, sonno, apporto calorico, fumo, consumo di alcol, frequenza cardiaca, pressione sanguigna, diabete, malattie cardiache.*


### Obiettivo dell'Analisi
L'obiettivo principale di questa analisi è sviluppare e valutare uno o più modelli di Machine Learning per **classificare** gli individui in base al loro rischio di sviluppare **malattie cardiache (`Heart_Disease`)**.
"""

# Importo librerie utili
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np

# Librerie relative ai modelli:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron

# Estraggo il file CSV salvato sul drive

# Link al CSV presente sul drive 
url = f"https://drive.google.com/uc?export=download&id=..."

try:
  # Creo una DataFrame a partire dal file
  df = pd.read_csv(url)
except Exception as e:
  # In caso di errore:
  print(f"Errore durante la creazione del df: {e}")

# Se è stata creata correttamente la df: Mostro a schermo l'head della tabella
df.head()

"""## Controllo ed Elaborazione dei dati"""

# Controllo se ci sono valori nulli, (True/False)
any_nulls = df.isnull().values.any()

# Nel caso ci siano dei valori nulli (True):
if any_nulls:
  print("Il DataFrame contiene valori nulli.")

  # Visualizzo, in base alla feature, quanti valori sono nulli (per capire se il DataFrame è utilizzabile oppure se mancano troppi dati)
  null_values = df.isnull().sum()
  print(null_values)

  # Rimuovo le righe che contengono valori nulli
  df = df.dropna()
  print("\nSono stati rimossi i valori nulli nel DataFrame")

else:
  # Se non ci sono valori nulli nel df
  print("Il DataFrame non contiene valori nulli.")

# Alcune colonne presentano delle variabini non numeriche: ['Gender', 'Smoker', 'Diabetic', 'Heart_Disease'],
# creo quindi delle nuove colonne con valori binari (0= No/Male, 1= Yes/Famale).
# Per indicare il genere:
df["Gender_bin"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(int)
# Per indicare chi fuma:
df["Smoker_bin"] = df["Smoker"].map({"No": 0, "Yes": 1}).astype(int)
# Per indiacare chi ha il diabete:
df["Diabetic_bin"] = df["Diabetic"].map({"No": 0, "Yes": 1}).astype(int)
# Per indicare chi ha disturbi cardiaci:
df["Heart_Disease_bin"] = df["Heart_Disease"].map({"No": 0, "Yes": 1}).astype(int)

# Divido le variabili relative alla Blood_Pressure in "massima"(="Systolic_Blood_Pressure") e "minima"(="Diastolic_Blood_Pressure")
df[["Systolic_Blood_Pressure", "Diastolic_Blood_Pressure"]] = df["Blood_Pressure"].str.split("/", expand=True).astype(int)
df[["Blood_Pressure", "Systolic_Blood_Pressure", "Diastolic_Blood_Pressure"]].head()

# Stampo tutti i nomi delle colonne
print(f"Nomi delle colonne: {df.columns}")

df.head()

"""## Predictive Modeling"""

# Selezionio solo le features che voglio usare per i modelli
features = ['Age', 'BMI', 'Daily_Steps',
        'Calories_Intake', 'Hours_of_Sleep', 'Heart_Rate',
        'Exercise_Hours_per_Week', 'Alcohol_Consumption_per_Week',
        'Gender_bin', 'Smoker_bin', 'Diabetic_bin', 'Systolic_Blood_Pressure',
        'Diastolic_Blood_Pressure']

X = df[features]
y = df['Heart_Disease_bin'] # La colonna target binaria


# Divido X e y in set di addestramento (80%) e di test (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y) # stratify=y per l'imbalancement


# Inizializzazione di oggetto StandardScaler
# Questo oggetto verrà usato per trasformare i dati in modo che abbiano media 0 e deviazione standard 1.
scaler = StandardScaler()
# Usiamo lo scaler SOLO sui dati di training (X_train).
# Lo scaler calcola la media e la dev. std. di ogni feature in X_train.
scaler.fit(X_train)

# Applichiamo la trasformazione (scaling) ai dati di training usando i parametri calcolati da X_train.
X_train_scaled = scaler.transform(X_train)

# Applichiamo la stessa trasformazione (usando i parametri di X_train) ai dati di test.
# Questo simula come tratteremmo nuovi dati "sconosciuti" in produzione.
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---

# Inizializzo il modello di Logistic Regression
model_lr = LogisticRegression(max_iter=1000)

# Alleno il modello sui dati di Training
model_lr.fit(X_train_scaled, y_train)

# Valutiamo la Logistic Regression:
y_pred_lr = model_lr.predict(X_test_scaled)
y_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1] # Prendo le probabilità per la classe positiva (1)
print("--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)*100}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

"""# Problema!

Noto subito che qualcosa non va nella classificazione, anche se l'`Accuracy` raggiunge un buon 90.0% (circa). Il problema diventa evidente guardando l'`AUC-ROC`: è circa 0.5 (o 50%), un valore molto basso. Questo significa che la capacità del modello di *distinguere* tra chi ha problemi cardiaci (classe 1) e chi non li ha (classe 0) è praticamente casuale, praticamente come tirare una moneta.

Questo succede a causa di uno **sbilanciamento delle classi** nel dataset: ci sono semplicemente troppi pochi esempi di persone con `Heart_Disease` uguale a 1. Di conseguenza, i modelli (con le impostazioni di default) faticano a imparare a riconoscere correttamente questa classe minoritaria.

La soluzione ideale sarebbe raccogliere più dati, specialmente per le persone con problemi cardiaci (`Heart_Disease` = 1 / "Yes"), per rendere il dataset più bilanciato.

Dato che non abbiamo altri dati, la strategia alternativa è quella di intervenire sui modelli o sui dati di addestramento. Possiamo usare tecniche per **bilanciare i pesi**, facendo in modo che gli errori sulla classe minoritaria (i pochi esempi con y=1) "pesino di più" durante l'allenamento del modello, forzandolo a prestarvi maggiore attenzione.
"""

# --- Logistic Regression con Bilanciamento dei pesi ---

# Inizializzo il modello di Logistic Regression
# Aggiungo {class_weight='balanced'} per far "pasare" di più le variabili che interessano a noi,
# che attualmente sono in netta minoranza.
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')

# Alleno il modello sui dati di Training
model_lr.fit(X_train_scaled, y_train)

# Valutiamo la Logistic Regression:
y_pred_lr = model_lr.predict(X_test_scaled)
y_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1] # Prendo le probabilità per la classe positiva (1)
print("--- Logistic Regression con Bilaciamento dei Pesi---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)*100}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

"""Possiamo notare come l'`Accuracy` sia scesa drasticamente, questo indica come effetticamente il modello predica i dati, ma ovviamente una soluzione del genere non va bene, essendo molto bassa la percentuale di valori classificati bene.


Provo ad utilizzare altri modelli.
"""

# --- Random Forest ---

# Inizializzo il modello di Random Forest
model_rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Alleno il modello sui dati di Training
model_rf.fit(X_train_scaled, y_train) # Uso comunque X_train_scaled anche se Random Forest è meno sensibile allo scaling

# Valutiamo il modello creato:
y_pred_rf = model_rf.predict(X_test_scaled)
y_proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1] # Prendo le probabilità per la classe positiva (1)
print("--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)*100}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# --- Support Vector Machine ---

# Inizializzo il modello Support Vector Machine
# AGGIUNGO: class_weight='balanced' per gestire lo sbilanciamento
# AGGIUNGO: probability=True per poter calcolare AUC-ROC dopo
model_svc = SVC(class_weight='balanced', probability=True, random_state=42)

# Alleno il modello sui dati di Training
model_svc.fit(X_train_scaled, y_train)

# Valutiamo il modello creato:
y_pred_svc = model_svc.predict(X_test_scaled)
y_proba_svc = model_svc.predict_proba(X_test_scaled)[:, 1] # Prendo le probabilità per la classe positiva (1)
print("--- Support Vector Machine ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svc)*100}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_svc):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svc))

# --- XGBoost Classifier ---

# Calcola prima il peso per bilanciare le classi
# Questo va fatto PRIMA di inizializzare il modello se usi scale_pos_weight
count_neg = np.sum(y_train == 0)
count_pos = np.sum(y_train == 1)
if count_pos > 0:
    scale_pos_weight_value = count_neg / count_pos
else:
    scale_pos_weight_value = 10 # Default se non ci sono positivi
# Stampiamo il valore calcolato come informazione utile
print(f"(Info preliminare: scale_pos_weight calcolato = {scale_pos_weight_value:.2f})")

# Inizializzo il modello XGBoost Classifier
# AGGIUNGO: scale_pos_weight con il valore calcolato per gestire lo sbilanciamento
# AGGIUNGO: random_state per riproducibilità
# Altri parametri spesso utili/necessari con XGBoost: use_label_encoder=False, eval_metric='logloss'
model_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight_value,
    eval_metric='logloss',    # Metrica usata internamente da XGBoost
    random_state=42
)

# Alleno il modello sui dati di Training
model_xgb.fit(X_train_scaled, y_train)

# Valutiamo il modello creato:
y_pred_xgb = model_xgb.predict(X_test_scaled)
y_proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1] # Prendo le probabilità per la classe positiva (1)
print("\n--- XGBoost Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

# --- Percettrone ---

# Inizializzo il modello Perceptron
# Aggiungo random_state per riproducibilità dell'inizializzazione casuale dei pesi
perceptron = Perceptron(random_state=42,class_weight='balanced')

# Alleno il modello sui dati di Training
perceptron.fit(X_train_scaled, y_train)


# Valutiamo il modello creato:
y_pred_perceptron = perceptron.predict(X_test_scaled)
print("--- Percettrone ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_perceptron)*100:.2f}%")
# AUC non calcolabile
print("\nClassification Report:")
print(classification_report(y_test, y_pred_perceptron))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_perceptron))

"""Un comportamento diverso è stato osservato con il **Percettrone**. Sebbene la sua Accuracy generale fosse bassa (52% circa) e la sua Precision sulla classe 1 molto scarsa (10% circa, indicando molti falsi positivi), questo modello è stato l'unico a raggiungere un **Recall per la classe 1 relativamente alto** (53% circa). Questo suggerisce che il Percettrone, in questa configurazione, ha "provato" più degli altri a identificare i casi positivi, riuscendoci circa la metà delle volte, anche se a costo di molti errori generali.

---

### Tentativo di Bilanciamento dei Dati con SMOTE

Come osservato nelle analisi precedenti, le tecniche di bilanciamento applicate direttamente ai modelli (come `class_weight` e `scale_pos_weight`) non hanno risolto in modo soddisfacente il problema dello sbilanciamento delle classi per questo dataset. Molti modelli hanno continuato a faticare nell'identificare la classe minoritaria (Heart_Disease = 1).

Un approccio differente, che possiamo considerare, consiste nel modificare direttamente la distribuzione dei dati di addestramento prima di fornirli al modello. Utilizzeremo la tecnica SMOTE (Synthetic Minority Over-sampling Technique) per generare campioni "sintetici" della classe minoritaria. L'obiettivo è creare un training set più bilanciato, sperando che i modelli possano apprendere pattern più robusti per entrambe le classi.

Applicheremo SMOTE solo ai dati di addestramento (`X_train_scaled`, `y_train`) e addestreremo nuovamente i modelli (questa volta senza i parametri `class_weight`/`scale_pos_weight`) sui dati risultanti (`X_train_smote`, `y_train_smote`). La valutazione verrà comunque effettuata sul set di test originale (`X_test_scaled`, `y_test`).
"""

from imblearn.over_sampling import SMOTE

print("Distribuzione classi prima di SMOTE:")
print(y_train.value_counts())

# Inizializza SMOTE
# random_state assicura che i risultati siano riproducibili
smote = SMOTE(random_state=42)

# Applica SMOTE ai dati di addestramento SCALATI
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\nDistribuzione classi dopo SMOTE:")
print(y_train_smote.value_counts())

# --- Ora usa X_train_smote e y_train_smote per addestrare i tuoi modelli ---

# --- Support Vector Machine con SMOTE ---

model_svc_smote = SVC(probability=True, random_state=42) # posso anche rimuovere class_weight='balanced', perchè non influenza più

# Alleno il modello sui dati di Training BILANCIATI con SMOTE
model_svc_smote.fit(X_train_smote, y_train_smote)

# Valutiamo il modello creato (sempre sul test set originale SCALATO)
y_pred_svc_smote = model_svc_smote.predict(X_test_scaled)
y_proba_svc_smote = model_svc_smote.predict_proba(X_test_scaled)[:, 1] # Prendo le probabilità per la classe positiva (1)
print("--- Support Vector Machine con SMOTE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svc_smote)*100:.2f}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_svc_smote):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svc_smote))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc_smote))

# --- Percettrone con SMOTE ---

# Inizializzo il modello Perceptron
perceptron_smote = Perceptron(random_state=42, class_weight='balanced')

# Alleno il modello sui dati di Training BILANCIATI con SMOTE
perceptron_smote.fit(X_train_smote, y_train_smote)

# Valutiamo il modello creato (sempre sul test set originale SCALATO)
y_pred_perceptron_smote = perceptron_smote.predict(X_test_scaled)

print("--- Percettrone con SMOTE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_perceptron_smote)*100:.2f}%")
# AUC non calcolabile direttamente con predict standard per Perceptron
print("\nClassification Report:")
print(classification_report(y_test, y_pred_perceptron_smote))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_perceptron_smote))

"""## Considerazioni Finali

L'addestramento di modelli standard (come la Regressione Logistica) con le impostazioni predefinite ha fatto scoprire alcune difficoltà: pur raggiungendo talvolta un'accuracy elevata, questa era fuorviante, in quanto i modelli fallivano quasi completamente nell'identificare i casi positivi, come indicato da AUC-ROC vicini a 0.5 e metriche nulle (Precision, Recall, F1-score) per la classe 1.

Per affrontare questo sbilanciamento, sono state esplorate due strategie principali:

1.  **Bilanciamento dei Pesi nel Modello:** Utilizzando parametri come `class_weight='balanced'` o `scale_pos_weight`, si è cercato di dare maggiore importanza agli errori sulla classe minoritaria durante l'addestramento. Questa tecnica ha prodotto risultati misti. Modelli come Random Forest e XGBoost hanno continuato a non riconoscere la classe 1. Il Percettrone ha mostrato un comportamento peculiare: pur con bassa accuracy (51.5%) e bassissima precision (10%), è riuscito a ottenere un Recall relativamente alto per la classe 1 (53%), suggerendo una tendenza a identificare i positivi, seppur a costo di molti falsi positivi. Anche SVM con `class_weight` ha mostrato scarse performance.

2.  **Bilanciamento dei Dati con SMOTE:** In alternativa, è stata applicata la tecnica SMOTE (Synthetic Minority Over-sampling Technique) per generare campioni sintetici della classe minoritaria direttamente nel set di addestramento. I modelli sono stati poi riaddestrati su questi dati bilanciati, senza utilizzare i parametri di pesatura delle classi. L'analisi si è concentrata su SVM e Percettrone:
    * **Support Vector Machine (SVM) con SMOTE:** Questo approccio ha mostrato un miglioramento rispetto alla versione con `class_weight`. L'Accuracy è salita al 79% e l'AUC-ROC è migliorato notevolmente a 0.6217. Anche l'F1-score per la classe 1 è leggermente aumentato (0.22). Sebbene la capacità di identificare correttamente i casi positivi rimanga limitata (Recall 0.32, Precision 0.17), il modello SVM con SMOTE offre un quadro complessivamente più bilanciato e performante rispetto ai tentativi precedenti.
    * **Percettrone con SMOTE:** Contrariamente all'SVM, il Percettrone ha mostrato un peggioramento nelle metriche chiave per la classe minoritaria quando addestrato con SMOTE. L'Accuracy generale è aumentata al 63%, ma il Recall per la classe 1 è crollato al 21% (dal 53% con `class_weight`) e la Precision è ulteriormente diminuita al 6%. L'F1-score per la classe 1 è sceso a 0.10. In questo caso specifico, SMOTE non è stato vantaggioso per il Percettrone nel contesto dell'identificazione dei casi positivi.

**Valutazione Complessiva:**

Un altra alternativa che abbiamo per cercare di riuscire a classificare Heart_Disease può essere quella di dividere il Training Set e il Testing Set secondo l'algoritmo TWIST [[link](https://www.researchgate.net/profile/Massimo-Buscema/publication/235693037_Training_with_Input_Selection_and_Testing_TWIST_Algorithm_A_Significant_Advance_in_Pattern_Recognition_Performance_of_Machine_Learning/links/0912f5129f9832ee6c000000/Training-with-Input-Selection-and-Testing-TWIST-Algorithm-A-Significant-Advance-in-Pattern-Recognition-Performance-of-Machine-Learning.pdf?origin=publication_detail&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uRG93bmxvYWQiLCJwcmV2aW91c1BhZ2UiOiJwdWJsaWNhdGlvbiJ9fQ)]. Sfortunatamente per questioni di tempo non posso implementare questa soluzione.


"""
