"""
preprocess.py - Modulo de preprocesamiento de texto para clasificacion de noticias falsas.

Este modulo contiene funciones para limpiar texto, cargar datos desde archivos CSV
y dividir el dataset en conjuntos de entrenamiento, validacion y prueba.

Autor: Curso de Modelado Predictivo 2026 - ESCOM IPN
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Stopwords en ingles (lista compacta para no depender de NLTK en produccion)
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now",
}


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Limpia una cadena de texto aplicando los siguientes pasos:
      1. Convierte a minusculas.
      2. Elimina URLs.
      3. Elimina signos de puntuacion y caracteres especiales.
      4. Elimina numeros aislados.
      5. Colapsa espacios multiples.
      6. (Opcional) Elimina stopwords en ingles.

    Parameters
    ----------
    text : str
        Texto crudo a limpiar.
    remove_stopwords : bool, optional
        Si es True, elimina las stopwords en ingles. Por defecto False.

    Returns
    -------
    str
        Texto limpio.

    Ejemplo
    -------
    >>> clean_text("  BREAKING!! Visit https://fake.com for more info.  ")
    'breaking visit for more info'
    >>> clean_text("Hello World!", remove_stopwords=True)
    'hello world'
    """
    if not isinstance(text, str):
        return ""

    # 1. Minusculas
    text = text.lower()

    # 2. Eliminar URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Eliminar puntuacion y caracteres especiales
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Eliminar numeros aislados (conserva palabras alfanumericas como "covid19")
    text = re.sub(r"\b\d+\b", "", text)

    # 5. Colapsar espacios multiples
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Stopwords (opcional)
    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
        text = " ".join(tokens)

    return text


def load_and_clean_data(filepath: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Carga un archivo CSV, maneja valores nulos, elimina duplicados y
    aplica la limpieza de texto a la columna indicada.

    Parameters
    ----------
    filepath : str
        Ruta al archivo CSV (por ejemplo, '../data/WELFake_Dataset.csv').
    text_col : str, optional
        Nombre de la columna que contiene el texto. Por defecto 'text'.
    label_col : str, optional
        Nombre de la columna que contiene la etiqueta. Por defecto 'label'.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio con columnas [text_col, label_col, 'clean_text'].

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    KeyError
        Si las columnas especificadas no existen en el CSV.
    """
    # Cargar CSV
    df = pd.read_csv(filepath)

    # Verificar que las columnas existan
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise KeyError(
                f"La columna '{col}' no existe en el archivo. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    # Conservar solo las columnas necesarias
    df = df[[text_col, label_col]].copy()

    # Eliminar filas con valores nulos en texto o etiqueta
    rows_before = len(df)
    df.dropna(subset=[text_col, label_col], inplace=True)
    rows_dropped_na = rows_before - len(df)
    if rows_dropped_na > 0:
        print(f"[INFO] Se eliminaron {rows_dropped_na} filas con valores nulos.")

    # Eliminar duplicados exactos
    rows_before = len(df)
    df.drop_duplicates(subset=[text_col], inplace=True)
    rows_dropped_dup = rows_before - len(df)
    if rows_dropped_dup > 0:
        print(f"[INFO] Se eliminaron {rows_dropped_dup} filas duplicadas.")

    # Asegurar que la etiqueta sea entera
    df[label_col] = df[label_col].astype(int)

    # Aplicar limpieza de texto
    df["clean_text"] = df[text_col].apply(clean_text)

    # Eliminar filas cuyo texto limpio quedo vacio
    df = df[df["clean_text"].str.len() > 0].copy()

    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Dataset final: {len(df)} filas.")

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    label_col: str = "label",
):
    """
    Divide un DataFrame en conjuntos de entrenamiento, validacion y prueba
    usando muestreo estratificado para mantener la proporcion de clases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos ya limpios.
    train_ratio : float
        Proporcion para entrenamiento (default 0.8).
    val_ratio : float
        Proporcion para validacion (default 0.1).
    test_ratio : float
        Proporcion para prueba (default 0.1).
    random_state : int
        Semilla para reproducibilidad (default 42).
    label_col : str
        Nombre de la columna de etiquetas (default 'label').

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        Si las proporciones no suman 1.0 (con tolerancia de 1e-9).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Las proporciones deben sumar 1.0, pero suman {total:.4f}."
        )

    # Primera division: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df[label_col],
    )

    # Segunda division: val vs test (dentro de temp_df)
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=temp_df[label_col],
    )

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df
