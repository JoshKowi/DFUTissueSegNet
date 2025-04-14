from utility import *

def train_semi_supervised_model(
        all_images_directory, 
        all_masks_directory, 
        accepted_pseudo_label_text = None, 
        accepted_pseudo_label_names = [],
        text_directory = DATA_PATH + 'texts/', 
        n_runs = 5):
    """Train a semi-supervised model using 50 pseudolabels and 5 runs
    Args:
        all_images_directory (str): Directory of all images
        all_masks_directory (str): Directory of all masks (important: needs to contain all labeled masks, the 50 accepted pseudolabels and the remaining new pseudo labels)
        current_pseudo_label_names (list): List of new pseudo label names
        accepted_pseudo_label_names (list, optional): List of accepted pseudo label names used for previous run. Defaults to [].
        text_directory (str): Path to the text files
        n_runs (int, optional): Number of runs. Defaults to 5.
    """
    # As default: Try to use all unlabeled images as pseudo-labels
    current_pseudo_label_names = read_names_ext(text_directory + 'raw_unsupervised_name.txt')

    if accepted_pseudo_label_text is not None:
        # Read the accepted pseudo labels from the text file
        accepted_pseudo_label_names = read_names(accepted_pseudo_label_text)
    current_pseudo_label_names = list(set(current_pseudo_label_names) - set(accepted_pseudo_label_names)) # remove accepted pseudo labels from the current pseudo labels
        
    supervised_label_names = read_names_ext(text_directory + 'labeled_train_names.txt')
    
    validation_names = read_names_ext(text_directory + 'labeled_val_names.txt')
    test_names = read_names_ext(text_directory + 'test_names.txt')

    x_train_dir = x_valid_dir = all_images_directory
    y_train_dir = y_valid_dir = all_masks_directory


    # The training CELL
    # ==========================================================================================================
    seeds = [random.randint(0, 5000) for _ in range(n_runs)] # generate 5 random seeds

    weight_factor = [1.0, 1.0, 1.0]

    save_dir_pred_root = DATA_PATH + 'predictions'
    os.makedirs(save_dir_pred_root, exist_ok = True)

    best_val_loss = float('inf')
    best_model_name = None

    for run, seed in enumerate(seeds):

        print('===================================================================')
        print('===================================================================')
        print(f'=========================== run {run} ============================')
        print('===================================================================')
        print('===================================================================')

        total_loss = base.HybridLoss(dice_loss, focal_loss, dce_loss, weight_factor)

        start = time.time() # start of training

        # Create a unique model name
        model_name = BASE_MODEL + '_padded_' + ENCODER + '_unsup50_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_seed_' + str(seed) + '_selfSupervised'
        print(model_name)

        aux_params=dict(
            classes=n_classes,
            activation=ACTIVATION,
            dropout=0.1, # dropout ratio, default is None
        )

        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            # aux_params=aux_params,
            classes=n_classes,
            activation=ACTIVATION,
            decoder_attention_type='pscse',
        )

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        model.to(DEVICE)

        # Optimizer
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        ])

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                    factor=0.1,
                                    mode='min',
                                    patience=10,
                                    min_lr=0.00001,
                                    verbose=True,
                                    )

        # seed = random.randint(0, 5000)

        seed = seeds[run]

        print(f'seed: {seed}')

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed) # seed for random number generator

        remaining_unsup_names_IDs = current_pseudo_label_names.copy() # make a copy of unsupervised remaining names
        random.shuffle(remaining_unsup_names_IDs) # shuffle unsupervised names
        remaining_unsup_names_IDs = remaining_unsup_names_IDs[:50] # take 50 unsupervised images

        list_IDs_train = supervised_label_names + remaining_unsup_names_IDs + accepted_pseudo_label_names # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< pay attention

        print('No. of training images: ', len(list_IDs_train))
        print('---> of these labeled: ', len(supervised_label_names))
        print('---> of these unlabeled: ', len(remaining_unsup_names_IDs))
        print('---> of these accepted pseudo labels: ', len(accepted_pseudo_label_names))
        print('No. of validation images: ', len(validation_names))
        print('No. of test images: ', len(test_names))

        # Save the randomly picked 50 unsupervised names in text files
        with open(os.path.join(text_directory, model_name + '_unsup_train.txt'), "w") as f:
            for name in remaining_unsup_names_IDs: print(name, file=f)

        # Checkpoint directory
        checkpoint_loc = DATA_PATH + 'checkpoints/' + model_name

        # Create checkpoint directory if does not exist
        if not os.path.exists(checkpoint_loc): os.makedirs(checkpoint_loc)

        # Dataloader ===================================================================
        train_dataset = Dataset(
            list_IDs_train,
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            to_categorical=TO_CATEGORICAL,
            resize=(RESIZE),
            n_classes=n_classes,
        )

        valid_dataset = Dataset(
            validation_names,
            x_valid_dir,
            y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            resize=(RESIZE),
            to_categorical=TO_CATEGORICAL,
            n_classes=n_classes,
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

        # create epoch runners =========================================================
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=total_loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=total_loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        # Train ========================================================================
        # train model for N epochs
        best_viou = 0.0
        best_vloss = 1_000_000.
        save_model = False # Initially start with False
        cnt_patience = 0

        store_train_loss, store_val_loss = [], []
        store_train_iou, store_val_iou = [], []
        store_train_dice, store_val_dice = [], []

        for epoch in range(EPOCHS):

            print('\nEpoch: {}'.format(epoch))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # Store losses and metrics
            train_loss_key = list(train_logs.keys())[0] # first key is for loss
            val_loss_key = list(valid_logs.keys())[0] # first key is for loss

            store_train_loss.append(train_logs[train_loss_key])
            store_val_loss.append(valid_logs[val_loss_key])
            store_train_iou.append(train_logs["iou_score"])
            store_val_iou.append(valid_logs["iou_score"])
            store_train_dice.append(train_logs["fscore"])
            store_val_dice.append(valid_logs["fscore"])

            # Track best performance, and save the model's state
            if  best_vloss > valid_logs[val_loss_key]:
                best_vloss = valid_logs[val_loss_key]
                print(f'Validation loss reduced. Saving the model at epoch: {epoch:04d}')
                cnt_patience = 0 # reset patience
                best_model_epoch = epoch
                save_model = True

            # Compare iou score
            elif best_viou < valid_logs['iou_score']:
                best_viou = valid_logs['iou_score']
                print(f'Validation IoU increased. Saving the model at epoch: {epoch:04d}.')
                cnt_patience = 0 # reset patience
                best_model_epoch = epoch
                save_model = True

            else: cnt_patience += 1

            # Learning rate scheduler
            scheduler.step(valid_logs[sorted(valid_logs.keys())[0]]) # monitor validation loss

            # Save the model
            if save_model:
                save(os.path.join(checkpoint_loc, 'best_model' + '.pth'),
                    epoch+1, model.state_dict(), optimizer.state_dict())
                save_model = False

            # Early stopping
            if EARLY_STOP and cnt_patience >= PATIENCE:
                print(f"Early stopping at epoch: {epoch:04d}")
                break

            # Periodic checkpoint save
            if not SAVE_BEST_MODEL:
                if (epoch+1) % PERIOD == 0:
                    save(os.path.join(checkpoint_loc, f"cp-{epoch+1:04d}.pth"),
                        epoch+1, model.state_dict(), optimizer.state_dict())
                    print(f'Checkpoint saved for epoch {epoch:04d}')

        if not EARLY_STOP and SAVE_LAST_MODEL:
            print('Saving last model')
            save(os.path.join(checkpoint_loc, 'last_model' + '.pth'),
                epoch+1, model.state_dict(), optimizer.state_dict())

        print('Best model epoch:', best_model_epoch)
        print('Min validation loss:', np.min(store_val_loss))
        min_val_loss = np.min(store_val_loss)

        end = time.time() # End of training

        print(f'Training time: {end - start:.2f} seconds')

        # Plot loss curves =============================================================
        fig, ax = plt.subplots(1,3, figsize=(12, 3))

        ax[0].plot(store_train_loss, 'r')
        ax[0].plot(store_val_loss, 'b')
        ax[0].set_title('Loss curve')
        ax[0].legend(['training', 'validation'])

        ax[1].plot(store_train_iou, 'r')
        ax[1].plot(store_val_iou, 'b')
        ax[1].set_title('IoU curve')
        ax[1].legend(['training', 'validation'])

        ax[2].plot(store_train_iou, 'r')
        ax[2].plot(store_val_iou, 'b')
        ax[2].set_title('Dice curve')
        ax[2].legend(['training', 'validation'])

        fig.tight_layout()

        save_fig_dir = DATA_PATH + "plots/"
        if not os.path.exists(save_fig_dir): os.makedirs(save_fig_dir)

        fig.savefig(os.path.join(save_fig_dir, model_name + '.png'))

        with open(DATA_PATH + 'plots/validation_scores.txt', "a") as f:
            f.write(f"{model_name} , {min_val_loss}\n")

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_model_name = model_name

    return best_model_name, best_val_loss
        
        



