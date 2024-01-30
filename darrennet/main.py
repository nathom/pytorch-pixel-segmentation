import click


@click.group()
def main():
    pass


@main.command()
def load_data():
    click.echo("Loading data...")


@main.command()
def train_network():
    click.echo("Training network...")


@main.command()
def run_inference():
    click.echo("Loading network and running inference...")


if __name__ == "__main__":
    main()
