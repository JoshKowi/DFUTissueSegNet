from utility import *
import matplotlib.pyplot as plt

# Create a function to read names from a text file, and add extensions
def read_names(txt_file, ext=".png"):
  with open(txt_file, "r") as f: names = f.readlines()

  names = [name.strip("\n") for name in names] # remove newline

  # Names are without extensions. So, add extensions
  names = [name + ext for name in names]

  return names


def execute_supervised_training():
    save_dir_pred_root = DATA_PATH + 'predictions/'
    os.makedirs(save_dir_pred_root, exist_ok = True)

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

    seed = random.randint(0, 5000)

    print(f'seed: {seed}')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    x_train_dir = x_valid_dir = DATA_PATH + 'Labeled/Padded/Images/TrainVal/'
    y_train_dir = y_valid_dir = DATA_PATH + 'Labeled/Padded/Annotations/TrainVal/'

    x_test_dir = DATA_PATH + 'Labeled/Padded/Images/Test/'
    y_test_dir = DATA_PATH + 'Labeled/Padded/Annotations/Test/'

    # Read train, test, and val names
    dir_txt = DATA_PATH + 'texts/'
    list_IDs_train = read_names(os.path.join(dir_txt, 'labeled_train_names.txt'), ext='.png')
    list_IDs_val = read_names(os.path.join(dir_txt, 'labeled_val_names.txt'), ext='.png')
    list_IDs_test = read_names(os.path.join(dir_txt, 'test_names.txt'), ext='.png')

    random.seed(seed) # seed for random number generator
    random.shuffle(list_IDs_train) # shuffle train names

    print('No. of training images: ', len(list_IDs_train))
    print('No. of validation images: ', len(list_IDs_val))
    print('No. of test images: ', len(list_IDs_test))

    # Create a unique model name
    model_name = BASE_MODEL + '_padded_aug_' + ENCODER + '_sup_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(model_name)

    # Default images
    DEFAULT_IMG_TRAIN = cv2.imread(os.path.join(x_train_dir, list_IDs_train[0]))[:,:,::-1]
    DEFAULT_MASK_TRAIN = cv2.imread(os.path.join(y_train_dir, list_IDs_train[0]), 0)
    DEFAULT_IMG_VAL = cv2.imread(os.path.join(x_valid_dir, list_IDs_val[0]))[:,:,::-1]
    DEFAULT_MASK_VAL = cv2.imread(os.path.join(y_valid_dir, list_IDs_val[0]), 0)

    # Checkpoint directory
    checkpoint_loc = DATA_PATH + 'checkpoints/' + model_name

    # Create checkpoint directory if does not exist
    if not os.path.exists(checkpoint_loc): os.makedirs(checkpoint_loc)

    # if SAVE_BEST_MODEL_ONLY: checkpoint_path = os.path.join(checkpoint_loc, 'best_model.pth')
    # else: checkpoint_path = os.path.join(checkpoint_loc, "cp-{epoch:04d}.pth")

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
        default_img=DEFAULT_IMG_TRAIN,
        default_mask=DEFAULT_MASK_TRAIN,
    )

    valid_dataset = Dataset(
        list_IDs_val,
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        resize=(RESIZE),
        to_categorical=TO_CATEGORICAL,
        n_classes=n_classes,
        default_img=DEFAULT_IMG_VAL,
        default_mask=DEFAULT_MASK_VAL,
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

    print(best_model_epoch)
    print('best valdiation loss = ' + str(best_vloss))

    # Plot loss curves =============================================================
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3, figsize=(12, 3))

    print(type(store_val_loss))
    print(store_val_loss)

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
        f.write(f"{model_name} , {best_vloss}\n")

    #del model
    #torch.cuda.empty_cache()
    return model_name, best_vloss


if __name__ == "__main__":
   model_name, best_vloss = execute_supervised_training()
   print('training_ended: ' + model_name + str(best_vloss))
