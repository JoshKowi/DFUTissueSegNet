from utility import *

prev2_model_name = 'MiT+pscse_padded_mit_b3_2025-03-27_13-17-55_selfSupervised'
prev_model_name = 'MiT+pscse_padded_mit_b3_unsup50_2025-03-28_16-15-29_seed_1028_selfSupervised'
phase='3'


def read_names(dir):
  """This function reads names from a text file"""
  with open(dir, "r") as f:
    names = f.readlines()
    names = [name.split('\n')[0] for name in names] # remove \n (newline)

  return names

# Create a function to read names from a text file, and add extensions
def read_names_ext(txt_file, ext=".png"):
  with open(txt_file, "r") as f: names = f.readlines()
  
  names = [name.strip("\n") for name in names] # remove newline
  # Names are without extensions. So, add extensions
  names = [name + ext for name in names]

  return names

def delt(dir, names):
  """This function deletes files specified in (names) from a directory (dir)"""
  for name in names:
    if os.path.exists(os.path.join(dir, name)): os.remove(os.path.join(dir, name))

def copy(dir_src, dir_dst, names):
  """This function copy files specified in (names) from (dir_src) to (dir_dst)"""
  for name in names:
    shutil.copy(os.path.join(dir_src, name), os.path.join(dir_dst, name))

#. Could restore the 50 predictions that were used for good training run after making new predictions on all images.
def replace(dir_ann_phase_prev, dir_ann_phase_current, prev_phase_names):
  """
  dir_ann_phase_prev: Director of annotations from previous training phase
  dir_ann_phase_current: Director of annotations of current training phase
  prev_phase_names: Names of previous training phase
  """
  delt(dir_ann_phase_current, prev_phase_names) # delete files from the train directory
  copy(dir_ann_phase_prev, dir_ann_phase_current, prev_phase_names) # replace annotations of current phase by the prev phase


# Image and label directories
x_train_dir = x_valid_dir = x_test_dir = DATA_PATH + 'current_composition/images'

# Backup annotation directory
y_train_dir1 = y_valid_dir1 = y_test_dir1 = DATA_PATH + 'all/masks'



dir_txt_save = DATA_PATH + 'texts/'
os.makedirs(dir_txt_save, exist_ok=True)

# Read unsupervised names
dir_txt = DATA_PATH + 'texts/'
unsup_names = read_names(os.path.join(dir_txt, 'Unsupervised_name.txt'))

# Read supervised train, test, and val names
sup_IDs_train = read_names_ext(os.path.join(dir_txt, 'labeled_train_names.txt'))
list_IDs_val = read_names_ext(os.path.join(dir_txt, 'labeled_val_names.txt'))
list_IDs_test = read_names_ext(os.path.join(dir_txt, 'test_names.txt'))



# Create new directory for this phase
y_train_dir = y_valid_dir = y_test_dir = DATA_PATH + 'current_composition/annotations_ph' + phase

os.makedirs(y_train_dir, exist_ok=True)
all_train_names = os.listdir(y_train_dir1)
copy(y_train_dir1, y_train_dir, all_train_names) # copy images from the backup train dir to current train dir

# Replace annotations of training dir by phase1 annotations
#. DIRECTORY OF 50 PREDICTIONS USED FOR GOOD TRAINING RUN PREVIOUS (NOT PHASE 1!)
dir_ann_phase1 = DATA_PATH + 'predictions/' + prev2_model_name + '_phase2'
dir_txt_phase1 = DATA_PATH + 'texts/' + prev_model_name + '_unsup_train.txt'

phase1_names = read_names(dir_txt_phase1) # read names
replace(dir_ann_phase1, y_train_dir, phase1_names)
print(f"removing {len(set(phase1_names))} names from previous run")
print(f"remaining_names: {len(set(unsup_names))}")
# Finding remaining names other than phase names
remaining_names = list(set(unsup_names) - set(phase1_names))
print(f"remaining_names after removal: {len(set(remaining_names))}")

# Replace existing unsupervised annotations in train dir by most recent phase annotations
dir_ann_phase2 = DATA_PATH + 'predictions/' + prev_model_name + '_phase2'
replace(dir_ann_phase2, y_train_dir, remaining_names)

n_runs = 1 # No. of runs

seeds = [random.randint(0, 5000) for _ in range(n_runs)] # generate 5 random seeds

weight_factor = [1.0, 1.0, 1.0]

save_dir_pred_root = DATA_PATH + 'predictions'
os.makedirs(save_dir_pred_root, exist_ok = True)

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

    remaining_unsup_names_IDs = remaining_names.copy() # make a copy of unsupervised remaining names
    random.shuffle(remaining_unsup_names_IDs) # shuffle unsupervised names
    remaining_unsup_names_IDs = remaining_unsup_names_IDs[:50] # take 50 unsupervised images

    list_IDs_train = sup_IDs_train + remaining_unsup_names_IDs + phase1_names # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< pay attention

    print('No. of training images: ', len(list_IDs_train))
    print('No. of validation images: ', len(list_IDs_val))
    print('No. of test images: ', len(list_IDs_test))

    # Save the randomly picked 50 unsupervised names in text files
    with open(os.path.join(dir_txt_save, model_name + '_unsup_train.txt'), "w") as f:
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
        list_IDs_val,
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
        metrics=my_metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=total_loss,
        metrics=my_metrics,
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

    # Save the model performance:
    stats_path = os.path.join(save_fig_dir, model_name + '_stats.txt')
    with open(stats_path, "w") as file:
        file.write(f"Model name: {model_name}\n")
        file.write(f"Best model epoch: {best_model_epoch}\n")
        file.write(f"Min validation loss: {np.min(store_val_loss)}\n")
        file.write(f"Training time: {end - start:.2f} seconds\n")
        file.write(f"Total training epochs: {len(store_train_loss)}\n")

        file.write("\nTrain loss:\n")
        for value in store_train_loss:
            file.write(f"{value}\n")
        file.write("\nVal loss:\n")
        for value in store_val_loss:
            file.write(f"{value}\n")

        file.write("\nTrain IoU:\n")
        for value in store_train_iou:
            file.write(f"{value}\n")
        file.write("\nVal IoU:\n")
        for value in store_val_iou:
            file.write(f"{value}\n")

        file.write("\nTrain dice:\n")
        for value in store_train_dice:
            file.write(f"{value}\n")
        file.write("\nVal dice:\n")
        for value in store_val_dice:
            file.write(f"{value}\n")

    print(f"Model performance saved at: {stats_path}")