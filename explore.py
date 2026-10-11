from llm_sdk import Small_LLM_Model
import json


def main():
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as cont:
        line_count = sum(1 for _ in cont)

    print(f"Número de líneas: {line_count}")

    with open(vocab_path, "r") as f:
        data = json.load(f)

    print("\nPrimeras 5 entradas:\n")

    if isinstance(data, dict):
        for i, (k, v) in enumerate(data.items()):
            if i >= 50:
                break
            print(f"{k} --> {v}")

    elif isinstance(data, list):
        for item in data[:50]:
            print(item)


if __name__ == "__main__":
    main()
