from utility import *

def generate_pseudo_labels(model_name, images_dir=None, images_list=None):
    """Generate pseudo labels for a given model and images."""
    
    if images_dir is None:
        print('No images directory given. Using default: ', DATA_PATH + 'Unlabeled/')
        images_dir = DATA_PATH + 'Unlabeled/'

    if images_list is None:
        images_list = os.listdir(images_dir)
    # Check if the images_dir contains the names of the images given in the list
    images_in_direcotry = os.listdir(images_dir)
    set_difference = set(images_list) - set(images_in_direcotry)
    if len(set_difference) > 0:
        print('The following images are not in the directory:', set_difference)
        print('Exiting...')
        exit()

    #create segmentation model with pretrained encoder
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


    # Checkpoint directory
    checkpoint_loc = DATA_PATH + 'checkpoints/' + model_name

    # Load model====================================================================
    checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



    x_test_dir = images_dir
    list_IDs_test = images_list
    print('No. of test images: ', len(list_IDs_test))



    # Test dataloader ==============================================================
    test_dataset = Dataset_without_masks(
        list_IDs_test,
        x_test_dir,
        preprocessing=get_preprocessing(preprocessing_fn),
        resize=(RESIZE),
        n_classes=n_classes,
    )

    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=6)

    # Prediction ===================================================================
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    import scipy.io as sio

    import warnings
    warnings.filterwarnings("ignore")

    save_pred = True
    threshold = 0.5
    ep = 1e-6
    raw_pred = []

    HARD_LINE = True

    # Save directory
    save_dir_pred = DATA_PATH + 'predictions/' + model_name + '_selfsupervised'
    if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

    save_dir_pred_palette = DATA_PATH + 'predictions_palette/' + model_name + '_selfsupervised'
    if not os.path.exists(save_dir_pred_palette): os.makedirs(save_dir_pred_palette)

    iter_test_dataloader = iter(test_dataloader)

    palette = [[128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    for enu, i in enumerate(range(len(list_IDs_test))):

        name = os.path.splitext(list_IDs_test[i])[0] # remove extension

        # Image-wise mean of metrics
        i_mp, i_mr, i_mdice, i_miou = [], [], [], []

        image = next(iter_test_dataloader) # get image and mask as Tensors

        pr_mask = model.predict(image.to(DEVICE)) # Move image tensor to gpu

        # Convert from onehot
        # gt_mask = torch.argmax(gt_mask_, dim=1)
        if TO_CATEGORICAL:
            pr_mask = torch.argmax(pr_mask, dim=1)

        # pr_mask = torch.argmax(pr_mask, dim=1)

        # Move to CPU and convert to numpy
        pred = pr_mask.squeeze().cpu().numpy()

        # Save raw prediction
        if RAW_PREDICTION: raw_pred.append(pred)

        # Save prediction as png
        if save_pred:
            "Uncomment for non-palette"
            cv2.imwrite(os.path.join(save_dir_pred, list_IDs_test[i]), np.squeeze(pred).astype(np.uint8))

            "Uncomment for palette"
            # Palette prediction
            pal_pred = np.squeeze(pred).astype(np.uint8)
            pal_pred = Image.fromarray(pal_pred)
            pal_pred = pal_pred.convert("P")
            pal_pred.putpalette(np.array(palette, dtype=np.uint8))

            # Store
            pal_pred.save(os.path.join(save_dir_pred_palette, list_IDs_test[i]))
    
    return save_dir_pred


if __name__ == "__main__":
    model_name = 'best_model'
    images_dir = DATA_PATH + 'Unlabeled/'
    images_list = os.listdir(images_dir)