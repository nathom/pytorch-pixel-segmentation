import random

import click

colors = [
    # "red",
    # "green",
    # "yellow",
    # "blue",
    # "magenta",
    # "cyan",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    # "bright_white",
]


def darren_print(text: str):
    words = text.split(" ")
    new_text = [click.style(w, fg=random.choice(colors)) for w in words]
    click.echo(" ".join(new_text))


def serious_darren_print(text: str):
    click.secho(text, fg=random.choice(colors))
