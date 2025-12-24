import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import re
import time
from tqdm import tqdm
import nlpaug.augmenter.word as naw

# -------------------------
# 1. Загрузка CSV и подготовка данных
# -------------------------
def load_csv_dataset(csv_path):
    """
    Загружает CSV с новостями.
    Использует только текст (без title) и преобразует метки:
    real -> 0, fake -> 1
    Также выводит статистику по меткам.
    """
    df = pd.read_csv(csv_path)

    # Используем только текст
    df['text'] = df['text'].fillna('')

    # Удаляем пустые тексты и метки
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"[ERROR] Датасет пуст после удаления пустых значений. Проверьте файл {csv_path}")

    # Считаем исходные метки
    original_counts = df['label'].value_counts()
    print("[INFO] Исходные метки:")
    print(original_counts.to_dict())

    # Преобразуем метки: real -> 0, fake -> 1
    # df['label'] = df['label'].map({'real': 0, 'fake': 1})
    # df = df.dropna(subset=['label']).reset_index(drop=True)
    # df['label'] = df['label'].astype(int)
    #
    # # Считаем метки после преобразования
    # transformed_counts = df['label'].value_counts()
    # print("[INFO] Метки после преобразования (0=real, 1=fake):", transformed_counts.to_dict())

    if df.empty:
        raise ValueError(f"[ERROR] Датасет пуст после преобразования меток. Проверьте столбец 'label' в {csv_path}")

    return df[['text', 'label']]


# -------------------------
# 2. Очистка текста
# -------------------------
def clean_text(text):
    """
    Приводит текст к нижнему регистру, удаляет URL и лишние символы
    """
    if isinstance(text, list):
        text = " ".join(map(str, text))

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)

    return text.strip()

# -------------------------
# 3. Аугментация fake новостей
# -------------------------
print("[INIT] Инициализация аугментатора WordNet...")
syn_aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.1)
print("[OK] Аугментатор готов")

def augment_text(text, n_aug=1):
    """
    Возвращает список аугментированных версий текста
    """
    augmented = [text]
    for _ in range(n_aug):
        try:
            aug_text = syn_aug.augment(text)
            if aug_text != text:
                augmented.append(aug_text)
        except Exception:
            pass
    return augmented

def augment_dataset(df, n_aug=1):
    """
    Применяет аугментацию только к fake новостям (label=1).
    Автоматически исправляет возможные проблемы с типами и пробелами в метках.
    """
    import time
    from tqdm import tqdm

    print(f"[STEP] Аугментация fake новостей (n_aug={n_aug})...")
    start = time.time()

    # --- Шаг 4: аугментация ---
    texts, labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        text, label = row["text"], row["label"]

        # Только fake новости
        if label == 1:
            augmented_texts = augment_text(text, n_aug)
            texts.extend(augmented_texts)
            labels.extend([1] * len(augmented_texts))
        else:
            texts.append(text)
            labels.append(0)

    print(f"[OK] Аугментация завершена за {time.time() - start:.1f} сек")
    return pd.DataFrame({"text": texts, "label": labels})



# -------------------------
# 4. PyTorch Dataset
# -------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_length=256):
        self.texts = [clean_text(t) for t in texts]
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# -------------------------
# 5. Создание DataLoader
# -------------------------
def create_dataloaders(df, batch_size=16, val_split=0.2, tokenizer_name='bert-base-uncased', max_length=256):
    dataset = NewsDataset(df['text'].tolist(), df['label'].tolist(), tokenizer_name, max_length)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
