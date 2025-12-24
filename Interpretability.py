import torch
from transformers import AutoTokenizer
from Model import TransformerClassifier

MODEL_PATH = r"/Graphics/fake_news_transformer.pt"
TOKENIZER_NAME = "bert-base-uncased"

def token_importance(model, tokenizer, text, device, max_length=512):
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Правильный способ для leaf-тензора
    embeddings = model.embedding(input_ids).detach()
    embeddings.requires_grad_(True)

    # forward
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs

    # предсказанный класс
    pred_class = torch.argmax(logits, dim=-1)
    score = logits[0, pred_class]

    model.zero_grad()
    score.backward(retain_graph=True)

    # теперь grads точно будет
    grads = embeddings.grad.abs().sum(dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    tokens, grads = merge_subtokens(tokens, grads)

    # tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    ignore_tokens = ["[PAD]", "[CLS]", "[SEP]", "s", "u", '"', ".", ","]
    results = [(t, g.item()) for t, g in zip(tokens, grads) if t not in ignore_tokens]

    # results = [(token, score.item()) for token, score in zip(tokens, grads)
    #            if token not in ["[PAD]", "[CLS]", "[SEP]"]]

    return results



def RunFeatureAnalysis():
    print("[INFO] Загрузка модели и tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    vocab_size = tokenizer.vocab_size
    model = TransformerClassifier(vocab_size=vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Модель загружена.")

    print("Введите текст для анализа (пустая строка — выход):\n")
    while True:
        text = input("> ").strip()
        if text == "":
            break

        # Определяем к какому классу модель относит текст
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_class_idx = torch.argmax(logits, dim=1).item()
            pred_class = "Fake" if pred_class_idx == 1 else "Real"

        print(f"\n[INFO] Модель относит текст к классу: {pred_class}\n")

        # Получаем наиболее значимые токены
        scores = token_importance(model, tokenizer, text, device)

        print("[RESULT] Топ-15 наиболее значимых токенов:")
        for token, score in sorted(scores, key=lambda x: x[1], reverse=True)[:15]:
            print(f"{token:15s} {score:.4f}")

        print("\nВведите следующий текст или Enter для выхода.")


def merge_subtokens(tokens, scores):
    merged_tokens = []
    merged_scores = []
    buffer = ""
    buffer_score = 0.0
    count = 0

    for token, score in zip(tokens, scores):
        if token.startswith("##"):
            buffer += token[2:]
            buffer_score += score
            count += 1
        else:
            if buffer:
                merged_tokens.append(buffer)
                merged_scores.append(buffer_score / count)
                buffer = ""
                buffer_score = 0.0
                count = 0
            buffer = token
            buffer_score = score
            count = 1

    if buffer:
        merged_tokens.append(buffer)
        merged_scores.append(buffer_score / count)

    return merged_tokens, merged_scores
