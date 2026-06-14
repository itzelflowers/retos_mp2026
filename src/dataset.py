"""
dataset.py - Pipeline de texto para el Clasificador de Noticias Falsas
======================================================================

Este modulo transforma texto crudo en tensores listos para PyTorch:

    texto -> tokens -> indices -> tensor (pad/trunc) -> DataLoader

Funciones principales:
    - build_vocabulary:   Construye el mapeo palabra -> indice
    - text_to_indices:    Convierte una cadena de texto a secuencia de indices
    - load_glove:         Carga embeddings preentrenados GloVe
    - FakeNewsDataset:    Dataset de PyTorch para noticias
    - create_dataloaders: Genera DataLoaders de entrenamiento, validacion y prueba

Curso: Modelado Predictivo 2026 - ESCOM, IPN
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────
# Indices especiales reservados en el vocabulario
# ──────────────────────────────────────────────
PAD_IDX = 0  # Relleno (padding) para completar secuencias cortas
UNK_IDX = 1  # Indice para palabras fuera del vocabulario (desconocidas)


# ──────────────────────────────────────────────
# 1. Construccion del vocabulario
# ──────────────────────────────────────────────
def build_vocabulary(texts: List[str], max_vocab: int = 20_000) -> Dict[str, int]:
    """Construye un mapeo palabra -> indice a partir de una lista de textos.

    El vocabulario se limita a las *max_vocab* palabras mas frecuentes.
    Las posiciones 0 y 1 estan reservadas para <PAD> y <UNK>.

    Parametros
    ----------
    texts : List[str]
        Lista de textos (ya limpios / preprocesados).
    max_vocab : int
        Numero maximo de palabras en el vocabulario (sin contar PAD/UNK).

    Retorna
    -------
    word2idx : Dict[str, int]
        Diccionario {palabra: indice}.
    """
    # Contar frecuencias de todas las palabras
    counter: Counter = Counter()
    for text in texts:
        if isinstance(text, str):
            tokens = text.lower().split()
            counter.update(tokens)

    # Tokens especiales ocupan las primeras posiciones
    word2idx: Dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}

    # Agregar las palabras mas frecuentes (hasta max_vocab)
    for word, _freq in counter.most_common(max_vocab):
        word2idx[word] = len(word2idx)

    return word2idx


# ──────────────────────────────────────────────
# 2. Conversion de texto a indices
# ──────────────────────────────────────────────
def text_to_indices(
    text: str,
    word2idx: Dict[str, int],
    max_len: int = 200,
) -> List[int]:
    """Tokeniza un texto y lo convierte en una secuencia de indices numericos.

    - Palabras fuera del vocabulario reciben el indice UNK_IDX.
    - Secuencias mas cortas que *max_len* se rellenan con PAD_IDX (padding).
    - Secuencias mas largas que *max_len* se recortan (truncado).

    Parametros
    ----------
    text : str
        Texto de entrada (una noticia).
    word2idx : Dict[str, int]
        Vocabulario generado por ``build_vocabulary``.
    max_len : int
        Longitud fija de la secuencia de salida.

    Retorna
    -------
    indices : List[int]
        Secuencia de longitud exacta *max_len*.
    """
    if not isinstance(text, str):
        # Si el texto es NaN u otro tipo, devolver solo padding
        return [PAD_IDX] * max_len

    tokens = text.lower().split()

    # Convertir cada token a su indice (o UNK si no existe)
    indices = [word2idx.get(token, UNK_IDX) for token in tokens]

    # Truncado: cortar si excede max_len
    indices = indices[:max_len]

    # Padding: rellenar con PAD_IDX si es mas corto que max_len
    padding_needed = max_len - len(indices)
    indices = indices + [PAD_IDX] * padding_needed

    return indices


# ──────────────────────────────────────────────
# 3. Carga de embeddings GloVe
# ──────────────────────────────────────────────
def load_glove(
    glove_path: str,
    word2idx: Dict[str, int],
    embed_dim: int = 50,
) -> np.ndarray:
    """Carga vectores GloVe y construye una matriz de embeddings alineada al vocabulario.

    Para cada palabra del vocabulario que aparezca en GloVe, se copia su vector.
    Las palabras sin vector en GloVe se inicializan aleatoriamente con distribucion
    normal (media=0, desviacion=0.6).

    Parametros
    ----------
    glove_path : str
        Ruta al archivo GloVe (p. ej. ``glove.6B.50d.txt``).
    word2idx : Dict[str, int]
        Vocabulario del proyecto.
    embed_dim : int
        Dimension de los vectores GloVe (50, 100, 200 o 300).

    Retorna
    -------
    embedding_matrix : np.ndarray
        Matriz de forma ``(len(word2idx), embed_dim)``.
    """
    vocab_size = len(word2idx)

    # Inicializar la matriz con valores aleatorios pequenos
    # (asi las palabras sin vector GloVe tendran una representacion razonable)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    # El vector de PAD debe ser ceros (no aporta informacion)
    embedding_matrix[PAD_IDX] = np.zeros(embed_dim)

    # Leer el archivo GloVe linea por linea
    palabras_encontradas = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == embed_dim:
                    embedding_matrix[idx] = vector
                    palabras_encontradas += 1

    cobertura = palabras_encontradas / vocab_size * 100
    print(
        f"GloVe cargado: {palabras_encontradas:,} / {vocab_size:,} palabras "
        f"encontradas ({cobertura:.1f}% de cobertura)"
    )

    return embedding_matrix


# ──────────────────────────────────────────────
# 4. Dataset de PyTorch
# ──────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    """Dataset de PyTorch para el clasificador de noticias falsas.

    Cada elemento devuelve una tupla ``(text_tensor, label_tensor)``
    donde *text_tensor* es la secuencia de indices (LongTensor) y
    *label_tensor* es la etiqueta 0/1 (FloatTensor).

    Parametros
    ----------
    texts : List[str]
        Lista de textos (noticias).
    labels : List[int]
        Etiquetas correspondientes (0 = real, 1 = falsa).
    word2idx : Dict[str, int]
        Vocabulario palabra -> indice.
    max_len : int
        Longitud fija de cada secuencia.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        word2idx: Dict[str, int],
        max_len: int = 200,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Convertir texto a secuencia de indices con padding/truncado
        indices = text_to_indices(text, self.word2idx, self.max_len)

        text_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return text_tensor, label_tensor


# ──────────────────────────────────────────────
# 5. Creacion de DataLoaders
# ──────────────────────────────────────────────
def create_dataloaders(
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Tuple[List[str], List[int]],
    word2idx: Dict[str, int],
    batch_size: int = 64,
    max_len: int = 200,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crea DataLoaders para entrenamiento, validacion y prueba.

    Parametros
    ----------
    train_data : Tuple[List[str], List[int]]
        Tupla ``(textos, etiquetas)`` de entrenamiento.
    val_data : Tuple[List[str], List[int]]
        Tupla ``(textos, etiquetas)`` de validacion.
    test_data : Tuple[List[str], List[int]]
        Tupla ``(textos, etiquetas)`` de prueba.
    word2idx : Dict[str, int]
        Vocabulario palabra -> indice.
    batch_size : int
        Tamano del lote (batch).
    max_len : int
        Longitud fija de cada secuencia.

    Retorna
    -------
    train_loader, val_loader, test_loader : Tuple[DataLoader, DataLoader, DataLoader]
    """
    train_dataset = FakeNewsDataset(
        texts=train_data[0],
        labels=train_data[1],
        word2idx=word2idx,
        max_len=max_len,
    )
    val_dataset = FakeNewsDataset(
        texts=val_data[0],
        labels=val_data[1],
        word2idx=word2idx,
        max_len=max_len,
    )
    test_dataset = FakeNewsDataset(
        texts=test_data[0],
        labels=test_data[1],
        word2idx=word2idx,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # Mezclar datos de entrenamiento en cada epoca
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,   # No mezclar validacion (resultados reproducibles)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,   # No mezclar prueba
    )

    print(f"DataLoaders creados:")
    print(f"  Entrenamiento: {len(train_dataset):,} muestras -> {len(train_loader):,} batches")
    print(f"  Validacion:    {len(val_dataset):,} muestras -> {len(val_loader):,} batches")
    print(f"  Prueba:        {len(test_dataset):,} muestras -> {len(test_loader):,} batches")
    print(f"  Batch size: {batch_size} | Max len: {max_len}")

    return train_loader, val_loader, test_loader
