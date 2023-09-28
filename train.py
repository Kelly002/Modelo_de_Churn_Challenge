import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline

def train_model(data, parameters):
    # leitura e separacao da target
    X, y = data.drop('Churn', axis = 1), data['Churn']

    # balanceamneto dos dados
    smote = SMOTE(random_state = 42)

    # One hot encoder
    colunas_categoricas = ['internet_service', 'contract', 'payment_method']
    one_hot_enc = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', dtype=int),
         colunas_categoricas),
        remainder='passthrough'
        )

    # Gera o modelo
     pipe = ImblearnPipeline([
        ('one_hot_enc', one_hot_enc), 
        ('smote', smote), 
        ('rf', rf)
    ])
    
    # Ajustar o pipeline
    pipe.fit(X, y)
    
    return pipe

    # salva o modelo
    def save_model(model, filename):
        with open('/content/drive/MyDrive/Colab Notebooks/Challenge-Dados' + filename, 'wb') as file:
            pickle.dump(model, file)


    # faz a leitura dos dados e define os parametros
    if __name__ == '__main__':
    data = pd.read_csv('../churn.csv')
    parameters = {'bootstrap': True,
                'criterion': 'entropy',
                'max_depth': 8,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'n_estimators': 120}

    model = train_model(data, parameters)
    save_model(model, 'modelo_producao.pkl') 

