import argparse
import os
import pickle

from tensorflow.keras.models import load_model

from model_utils import generate_text, load_doc


def parse_args():
    parser = argparse.ArgumentParser(description="Load a trained LSTM model and generate text from a seed string.")
    parser.add_argument("--model-dir", required=True, help="Directory where the trained model and tokenizer are stored")
    parser.add_argument("--seed-text", required=True, help="Seed text to begin generation")
    parser.add_argument("--n-words", type=int, default=50, help="Number of words to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    model_path = os.path.join(model_dir, "model-best.keras")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    generated = generate_text(model, tokenizer, args.seed_text, n_words=args.n_words, seq_length=model.input_shape[1])
    print(generated)


if __name__ == "__main__":
    main()
