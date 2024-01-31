import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from . import theme


@click.group(
    cls=HelpColorsGroup,
    help_headers_color="blue",
    help_options_color="bright_red",
    help_options_custom_colors={"--help": "bright_green"},
)
def main():
    """Welcome to Darrennet, the world's most advanced CNN for pixel segmentation."""
    pass


@main.command(cls=HelpColorsCommand)
def load():
    theme.serious_darren_print("Loading data...")


@main.command(cls=HelpColorsCommand)
def train():
    theme.darren_print("Training network...")


@main.command(cls=HelpColorsCommand)
def run():
    theme.darren_print("Loading network and running inference...")


if __name__ == "__main__":
    main()
