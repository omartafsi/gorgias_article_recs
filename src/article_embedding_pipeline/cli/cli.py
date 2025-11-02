import logging

import click

from article_embedding_pipeline.cli.main_generate_embeddings import generate_embeddings


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(generate_embeddings)
    command_group()
