import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChangShuoRadioRecognition Performance Results"
    )
    parser.add_argument("config", help="performance config file path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start analyze
    runner.analyze()


if __name__ == "__main__":
    main()
