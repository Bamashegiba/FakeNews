from NewsParser import NewsPars
from FakeNewsGenerator import GenerateFakeNews
from Train import TrainFakeNewsClassifier

from Interpretability import RunFeatureAnalysis

OUTPUT_FILE_APNEWS="News/apnews_articles.jsonl"
OUTPUT_FILE_LLAMA_MODEL = "News/fake_news_LLAMA_MODEL.jsonl"
OUTPUT_FILE_QWEN_MODEL = "News/fake_news_QWEN_MODEL.jsonl"

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():

    while True:
        print("\n=== Fake News Project ===")
        print("1. Классификация новостей")
        print("2. Обучение модели")
        print("3. Парсинг новостей")
        print("4. Генерация фейковых новостей")
        print("5. Наиболее значимые признаки")
        print("6. Выход")

        choice = input("Выберите пункт меню: ").strip()
        if choice == "1":
            print("[INFO] Классификация новостей пока не реализовано (заглушка).")

        elif choice == "2":
            print("[INFO] Запуск обучения модели классификатора.")
            TrainFakeNewsClassifier()
            # TrainFakeNewsClassifier(
            #     news_dir="News",
            #     epochs=10,
            #     batch_size=8,
            #     max_length=256,
            #     lr=2e-4
            # )

        elif choice == "3":
            print("[INFO] Запуск парсинга новостей.")
            NewsPars(output_file=OUTPUT_FILE_APNEWS)

        elif choice == "4":
            fake_news_generator_menu()

        elif choice == "5":
            print("[INFO] Анализ наиболее значимых признаков.")
            RunFeatureAnalysis()

        elif choice == "6":
            print("[INFO] Завершение работы программы.")
            break


        else:
            print("[WARN] Некорректный ввод. Пожалуйста, выберите пункт от 1 до 5.")

def fake_news_generator_menu():
    print("\n=== Выберите модель для генерации ===")
    print("1. LLAMA_MODEL")
    print("2. QWEN_MODEL")
    print("3. Выход")

    choice = input("Выберите пункт меню: ").strip()

    num_articles = int(input("Введите количество новостей для генерации: "))

    if choice == "1":
        # GenerateFakeNews("llama", NUM_ARTICLES_PER_MODEL=5, output_file="fake_news_llama.jsonl")
        GenerateFakeNews(model_type="llama", NUM_ARTICLES_PER_MODEL=num_articles, output_file=OUTPUT_FILE_LLAMA_MODEL)

    elif choice == "2":
        GenerateFakeNews(model_type="qwen", NUM_ARTICLES_PER_MODEL=num_articles, output_file=OUTPUT_FILE_QWEN_MODEL)

    elif choice == "3":
        print("[INFO] Выход в главное меню...")



if __name__ == "__main__":
    main()