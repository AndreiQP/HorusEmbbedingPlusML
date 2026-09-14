import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


def parse_model_filename(fname):
    # expected: {origin}_{path_data}_{embedding}_{ModelName}.joblib
    base = os.path.basename(fname)
    if base.endswith('.joblib'):
        base = base[:-7]

    known_origins = ['augmented', 'external_only']
    origin = None
    for known in known_origins:
        if base.startswith(known + '_'):
            origin = known
            base = base[len(known) + 1:]
            break

    if origin is None:
        return None, None, None, None

    parts = base.split('_')
    if len(parts) < 3:
        return None, None, None, None

    embedding = parts[-2]
    model_name = parts[-1]
    path_data = '_'.join(parts[:-2])
    return origin, path_data, embedding, model_name


def load_validation_split(path_data, embedding, origin):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    train_dir = os.path.join(repo_root, 'datasets', 'dataset_train', 'processed', path_data)
    parquet_file = os.path.join(train_dir, f"{embedding}_chunks.parquet")
    if not os.path.exists(parquet_file):
        print(f"Train file not found for {path_data}/{embedding}: {parquet_file}")
        return None, None

    df = pd.read_parquet(parquet_file)
    if origin == 'external_only' and 'origin' in df.columns:
        df = df[df['origin'] == 'external']
    if len(df) == 0:
        print(f"Nenhum dado carregado para {origin}/{path_data}/{embedding}")
        return None, None

    X = np.array(list(df['text_embedded']))
    y = np.array(list(df['label']))

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_val, y_val


def evaluate_model_on_validation(model_path):
    origin, path_data, embedding, model_name = parse_model_filename(model_path)
    if origin is None:
        print(f"Nome de modelo inesperado: {model_path}")
        return None

    model = joblib.load(model_path)
    X_val, y_val = load_validation_split(path_data, embedding, origin)
    if X_val is None:
        return None

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fp_total = int(fp)
        fn_total = int(fn)
        tp_total = int(tp)
        tn_total = int(tn)
    else:
        tp_total = int(np.trace(cm))
        fp_total = int(np.sum(np.sum(cm, axis=0) - np.diag(cm)))
        fn_total = int(np.sum(np.sum(cm, axis=1) - np.diag(cm)))
        tn_total = int(cm.sum() - tp_total - fp_total - fn_total)

    fp_fn_sum = fp_total + fn_total

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    return {
        'origin': origin,
        'path_data': path_data,
        'embedding': embedding,
        'model_name': model_name,
        'train_type': f"{origin}_{path_data}",
        'label': f"{origin}_{path_data}_{embedding}_{model_name}",
        'accuracy': round(acc, 3),
        'f1_macro': round(f1_macro, 3),
        'precision_macro': round(precision_macro, 3),
        'recall_macro': round(recall_macro, 3),
        'tp': tp_total,
        'tn': tn_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'fp_fn_sum': fp_fn_sum,
        'confusion_matrix': cm,
        'classification_report': report,
    }


def evaluate_all_models(models_dir=None):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if models_dir is None:
        models_dir = os.path.join(repo_root, 'trained_models')

    entries = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith('.joblib'):
            continue
        model_path = os.path.join(models_dir, fname)
        res = evaluate_model_on_validation(model_path)
        if res is None:
            continue
        entries.append(res)

    if not entries:
        print('Nenhum modelo avaliado. Verifique caminhos e arquivos.')
        return pd.DataFrame()

    return pd.DataFrame(entries)


def plot_metrics(df_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df_metrics_sorted = df_metrics.sort_values('f1_macro', ascending=False)
    labels = df_metrics_sorted['label']

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    plt.bar(x - 0.15, df_metrics_sorted['accuracy'], width=0.3, label='Accuracy')
    plt.bar(x + 0.15, df_metrics_sorted['f1_macro'], width=0.3, label='F1-macro')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('Score')
    plt.title('Accuracy and F1-macro por modelo')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_f1_comparison.png'))
    plt.close()


def plot_confusion_matrix(cm, labels, out_path, title=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Pred')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(models_dir=None, out_dir=None):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if models_dir is None:
        models_dir = os.path.join(repo_root, 'trained_models')
    if out_dir is None:
        out_dir = os.path.join(repo_root, 'machine_learning', 'visualizations')

    os.makedirs(out_dir, exist_ok=True)

    df_metrics = evaluate_all_models(models_dir=models_dir)
    if df_metrics.empty:
        return

    df_metrics.to_csv(os.path.join(out_dir, 'models_metrics_summary.csv'), index=False)
    plot_metrics(df_metrics, out_dir)

    for _, row in df_metrics.iterrows():
        cm = row['confusion_matrix']
        report = row['classification_report']
        lab = row['label']
        cm_path = os.path.join(out_dir, f'cm_{lab}.png')
        class_labels = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
        if not class_labels:
            n = row['confusion_matrix'].shape[0]
            class_labels = list(map(str, range(n)))
        plot_confusion_matrix(cm, class_labels, cm_path, title=f'CM: {lab}')
        rep_df = pd.DataFrame(report).transpose()
        rep_df.to_csv(os.path.join(out_dir, f'report_{lab}.csv'))

    df_metrics.to_csv(os.path.join(out_dir, 'models_metrics_summary_full.csv'), index=False)
    print(f'Visualizations and metrics saved em: {out_dir}')


if __name__ == '__main__':
    main()


