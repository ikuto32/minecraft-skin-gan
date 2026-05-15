from src.config import parse_train_args
from src.engine import train


def main() -> None:
    config = parse_train_args()
    train(config)


if __name__ == "__main__":
    main()
