import argparse
import os

from model_utils import build_training_data, clean_lines, define_model, generate_text, load_doc, save_model_artifacts, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM text model from a .txt corpus.")
    parser.add_argument("--input", required=True, help="Path to the .txt input file")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained model and tokenizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input text file not found: {input_path}")
    if not input_path.lower().endswith(".txt"):
        raise ValueError(f"Input file must be a .txt file: {input_path}")
    if os.path.getsize(input_path) == 0:
        raise ValueError(f"Input text file is empty: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    doc_text = load_doc(input_path)
    lines = clean_lines(doc_text)
    X, y, tokenizer, seq_length, vocab_size = build_training_data(doc_text)
    model = define_model(vocab_size, seq_length)
    train_model(model, X, y, output_dir, epochs=args.epochs, batch_size=args.batch_size)
    save_model_artifacts(model, tokenizer, output_dir)

    seed_text = lines[0]
    generated = generate_text(model, tokenizer, seed_text, n_words=30, seq_length=seq_length)
    print("Seed text:")
    print(seed_text)
    print("\nGenerated sample:")
    print(generated)

    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()
