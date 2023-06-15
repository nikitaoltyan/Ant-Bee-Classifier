import matplotlib.pyplot as plt
import os


def plot_training_data(model_history, number_of_runs, show_flag=False):
    plt.plot(model_history.history['precision'], label='Train precision')
    plt.plot(model_history.history['recall'], label='Train recall')
    plt.plot(model_history.history['val_precision'], label='Val precision')
    plt.plot(model_history.history['val_recall'], label='Val recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    if show_flag:
        plt.show()

    # Save file
    plt.savefig(f'../runs/run_{number_of_runs}/train_history.png', bbox_inches='tight')

