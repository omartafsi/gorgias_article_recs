import logging

import click

from reranker.cli.train import train


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(train)
    command_group()
