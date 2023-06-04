import click
from model import *
from process_data import *

from model import make_model

@click.command()
@click.option('--batch_size', '-b', type=int, default=32, help='Batch size of data')
@click.option('--epochs', '-e', type=int, default=5, help='Epochs of training')
@click.option('--val_split', '-v', type=float, default=0.1, help='Split of data for validation during training')
@click.option('--verbose', '-vb', type=int, default=0, help='Visualization flag for training')
@click.option('--history', '-h', type=bool, default=False, help='Training history flag for visualize & save result')
def train(batch_size, epochs, val_split, verbose, history):
    X_train, y_train = prepare_train_data()
    model = make_model(X_train.shape[1])

    # Fitting
    train_history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=val_split, verbose=verbose)

    # Save history data
    if history:
        pass
        # TODO
        # Visualize & save history data. Use utils.

    # Save the trained model
    model.save('trained_model.h5')


if __name__ == '__train__':
    train()