from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def get_classifier(model_name: str, length_data: int = None):
    """
    Recebe o nome do modelo em formato de string e retorna a instância 
    do algoritmo configurado e pronto para receber o .fit()
    """
    model_name = model_name.strip().upper()
    
    if model_name == "RF":
        if length_data is None:
            return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:   
            square = int(length_data ** 0.5)
            return RandomForestClassifier(n_estimators=square, n_jobs=-1, random_state=42)
        
    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
    elif model_name == "SVM":
        return SVC(kernel='linear', probability=True, random_state=42)
        
    elif model_name == "NAIVEBAYES":
        return Pipeline([
            ('scaler', MinMaxScaler()), 
            ('nb', MultinomialNB())
        ])
        
    elif model_name == "XGBOOST":
        if not HAS_XGBOOST:
            raise ImportError("xgboost não está instalado. Instale com: pip install xgboost")
        if length_data is None:
            return xgb.XGBClassifier(n_estimators=100, learning_rate=0.001, eval_metric='logloss', random_state=42)
        else: 
            square = int(length_data ** 0.5)
            return xgb.XGBClassifier(n_estimators=square, learning_rate=0.001, eval_metric='logloss', random_state=42)
        
    else:
        raise ValueError(
            f"Modelo '{model_name}' não suportado. "
            f"Escolha entre: RF, KNN, SVM, NAIVEBAYES, XGBOOST."
        )
    
def get_all_classifiers(length_data: int = None):
    """
    Retorna um dicionário com todas as instâncias dos classificadores disponíveis.
    """
    classifiers = {
        "RF": get_classifier("RF", length_data),
        "KNN": get_classifier("KNN", length_data),
        "SVM": get_classifier("SVM", length_data),
        "NAIVEBAYES": get_classifier("NAIVEBAYES", length_data),
    }
    if HAS_XGBOOST:
        classifiers["XGBOOST"] = get_classifier("XGBOOST", length_data)
    return classifiers