import random

import click

_serious: bool = False


def set_serious(s: bool):
    global _serious
    _serious = s


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


def print(text: str, *args, **kwargs):
    global _serious
    if _serious:
        serious_darren_print(text, *args, **kwargs)
    else:
        darren_print(text, *args, **kwargs)


def darren_print(text: str, *args, **kwargs):
    words = text.split(" ")
    new_text = [click.style(w, fg=random.choice(colors)) for w in words]
    click.echo(" ".join(new_text), *args, **kwargs)


def serious_darren_print(text: str, *args, **kwargs):
    click.echo(text, *args, **kwargs)
