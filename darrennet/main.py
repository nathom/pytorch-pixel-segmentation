import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from . import theme

@click.group(
    cls=HelpColorsGroup,
    help_headers_color="blue",
    help_options_color="bright_red",
    help_options_custom_colors={"--help": "bright_green"},
)
@click.option("-s", "--serious", help="Turn on Serious Darren mode.", is_flag=True)
def main(serious):
    """Welcome to DarrenNet, the world's most advanced CNN for pixel segmentation."""
    global print_function
    if serious:
        print_function = theme.serious_darren_print
    else:
        print_function = theme.darren_print


@main.command(cls=HelpColorsCommand)
def download():
    """Download and save the dataset."""
    print_function("Loading data...")


@main.command(cls=HelpColorsCommand)
def evolve():
    """Train the model."""
    print_function("Training network...")



@main.command(cls=HelpColorsCommand)
def insight():
    """Run inference on the model."""
    print_function("Loading network and running inference...")



if __name__ == "__main__":
    main()
