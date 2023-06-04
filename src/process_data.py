import click


@click.command()
@click.option('--data_num', '-n', type=int, default=-1, help='Number of data to sort')
def sort_data(data_num):
    # TODO:
    # Add here collecting data from raw folder and sorting it.
    print(data_num)
    pass
# def


def prepare_train_data():
    # TODO:
    # Add here data processing and augmentation
    # train_df, test_df =
    return None, None