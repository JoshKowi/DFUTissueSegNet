from datetime import datetime
from supervised_training import execute_supervised_training, DATA_PATH
import os
import shutil
from pseudo_label_generation import generate_pseudo_labels
from semisupervised_training import train_semi_supervised_model
from test_evaluation import evaluate_on_test_data
from utility import copy_all, replace, read_names
import matplotlib.pyplot as plt
import csv

def plot_training_stats(training_stats, starting_time=None):
    if starting_time is None:
        starting_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    val_losses = [entry['val_loss'] for entry in training_stats]
    test_scores = [entry['test_dice'] for entry in training_stats]
    labels = [entry['model_name'] for entry in training_stats]

    with open(DATA_PATH + 'plots/' + starting_time + '_training_stats.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model_name', 'val_loss', 'test_dice'])
            writer.writeheader()
            writer.writerows(training_stats)

    plt.figure(figsize=(10, 4))

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Achse 1: Val Loss (linke Y-Achse)
    ax1.plot(val_losses, marker='o', color='blue', label='Val Loss')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Validation Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)

    # Achse 2: Test Score (rechte Y-Achse)
    ax2 = ax1.twinx()
    ax2.plot(test_scores, marker='o', color='green', label='Test Score')
    ax2.set_ylabel('Test Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    fig.savefig(DATA_PATH + 'plots/' + starting_time + '_plot.png')
    plt.close()


def train_supervised_model(number_training_runs):
    
    best_model_name = None
    best_validation_loss = float("inf")

    for i in range(number_training_runs):
        model_name, model_val_loss = execute_supervised_training()
        
        if model_val_loss < best_validation_loss:
            best_model_name = model_name
            best_validation_loss = model_val_loss

    print('\n\n///////////////////////////SUPERVISED TRAINING FINISHED///////////////////////////////////////\n')
    print(f'Trained {number_training_runs} times\n')
    print(f'\nBest_Model: {best_model_name}\n')
    print(f'Validation Loss: {best_validation_loss*10}')

    return best_model_name, best_validation_loss*10


def copy_masks_to_wd(phase_name, pseudo_labels_directory, accepted_pseudo_labels_text=None, accepted_pseudo_labels_directory=None):
    # Copies all labels and pseudo-labels into a new directory

    mask_directory = DATA_PATH + 'working_directories/' + phase_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(mask_directory)

    # Copy the masks for labeled Images
    copy_all(DATA_PATH + 'Labeled/Padded/Annotations/TrainVal', mask_directory)

    # copy the masks for unlabeled images
    copy_all(pseudo_labels_directory, mask_directory)

    if accepted_pseudo_labels_text is not None:
        if accepted_pseudo_labels_directory is None:
            print('Accepted pseudo labels text file given, but no directory given. Exiting...')
            exit()
        # copy the accepted pseudo-labels to the mask directory
        accepted_pseudo_labels_list = read_names(accepted_pseudo_labels_text)
        replace(accepted_pseudo_labels_directory, mask_directory, accepted_pseudo_labels_list)
        print(f'Replaced {len(accepted_pseudo_labels_list)} accepted pseudo-labels in {mask_directory} with the same ones in {accepted_pseudo_labels_directory}')
        
    # Log how the masks are composed
    with open(mask_directory + '/content.txt', 'w') as f:
        f.write(f'Used pseudo labels from {pseudo_labels_directory}\n')
        f.write(f'Used accepted pseudo labels from {accepted_pseudo_labels_directory}\n\n')
        if accepted_pseudo_labels_text is not None:
            f.write('Accepted pseudo labels:\n')
            for name in accepted_pseudo_labels_list:
                f.write(name + '\n')

    return mask_directory


### WHOLE TRAINING PROCESS
def main():
    training_stats = []
    starting_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('Starting training process at', starting_time)

    # Train a supervised model 5 times, take the best one
    best_supervised_model_name, best_validation_loss = train_supervised_model(8)
    _, test_dice = evaluate_on_test_data(best_supervised_model_name)
    training_stats.append({'model_name': best_supervised_model_name, 'val_loss': best_validation_loss, 'test_dice': test_dice })

    # Let the best model generate pseudo labels from the unlabeled data
    pseudo_labels_from_supervised_model = generate_pseudo_labels(best_supervised_model_name)

    # Create working directory for masks for semi-supervised training
    mask_directory = copy_masks_to_wd('first_semisupervised_training', pseudo_labels_from_supervised_model)

    # Train a 5 models on the labeled data using 50 pseudo-labels, return the best one
    previous_validation_loss = best_validation_loss
    previous_best_model_name = best_supervised_model_name
    best_ss_model_name, best_validation_loss = train_semi_supervised_model(DATA_PATH + 'all/images', mask_directory)
    accepted_pseudo_labels_text = DATA_PATH + 'texts/' + best_ss_model_name + '_unsup_train.txt'
    accepted_pseudo_labels_directory = pseudo_labels_from_supervised_model

    _, test_dice = evaluate_on_test_data(best_ss_model_name)
    training_stats.append({ 'model_name': best_ss_model_name, 'val_loss': best_validation_loss, 'test_dice': test_dice })

    # Continue training until the validation loss does not improve
    i = 2
    while(i < 12):
        # Let the best model generate pseudo labels from the unlabeled data
        pseudo_labels_from_ss_model = generate_pseudo_labels(best_ss_model_name)

        # Create working directory for masks for semi-supervised training, Using the new pseudo labels and the 50 pseudo-labels from previous run
        mask_directory = copy_masks_to_wd(f'no{i}_semisupervised_training', pseudo_labels_from_ss_model, accepted_pseudo_labels_text, accepted_pseudo_labels_directory)

        # Train a 5 models on the labeled data using 50 pseudo-labels, return the best one
        previous_validation_loss = best_validation_loss
        previous_best_model_name = best_ss_model_name
        best_ss_model_name, best_validation_loss = train_semi_supervised_model(DATA_PATH + 'all/images', mask_directory, accepted_pseudo_labels_text)
        accepted_pseudo_labels_text = DATA_PATH + 'texts/' + best_ss_model_name + '_unsup_train.txt'
        accepted_pseudo_labels_directory = pseudo_labels_from_ss_model

        _, test_dice = evaluate_on_test_data(best_ss_model_name)
        training_stats.append({ 'model_name': best_ss_model_name, 'val_loss': best_validation_loss, 'test_dice': test_dice })

        i += 1
        


    # If the validation loss did not improve, stop training and evaluate the best model on test data
    print('\n\n///////////////////////////SEMI-SUPERVISED TRAINING FINISHED///////////////////////////////////////\n')
    print('There was no improvement in validation loss, stopping training')
    print(f'Best_Model: {previous_best_model_name}\n')
    print(f'Validation Loss: {previous_validation_loss}')
    print('Starting Test-Evaluation')

    # Evaluate the best model on test data
    prediction_directory, test_dice = evaluate_on_test_data(previous_best_model_name, save_pred=True)

    # Plot Training Stats
    plot_training_stats(training_stats, starting_time=starting_time)

    print('Evaluation finished')
    print(f'Predictions saved in {prediction_directory}')



if __name__ == "__main__":
    main()
