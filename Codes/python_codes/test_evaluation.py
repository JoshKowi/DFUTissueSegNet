from utility import *

import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import scipy.io as sio

import warnings
warnings.filterwarnings("ignore")


def get_model(model_name):
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        # aux_params=aux_params,
        classes=n_classes,
        activation=ACTIVATION,
        decoder_attention_type='pscse',
    )


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
    checkpoint_loc = DATA_PATH + '/checkpoints/' + model_name

    # =================================== Inference ================================
    # Load model====================================================================
    checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model


def evaluate_on_test_data(
        model_name,
        x_test_dir = DATA_PATH + 'Labeled/Padded/Images/Test/',
        y_test_dir = DATA_PATH + 'Labeled/Padded/Annotations/Test/',
        list_IDs_test = read_names_ext(DATA_PATH + 'texts/test_names.txt'),
        save_pred_ext = '_test',
        save_pred = False):
    threshold = 0.5
    ep = 1e-6
    raw_pred = []

    HARD_LINE = True

    model = get_model(model_name)

    # Test dataloader ==============================================================
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


    test_dataset = Dataset(
        list_IDs_test,
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        resize=(RESIZE),
        to_categorical=False, # don't convert to onehot now
        n_classes=n_classes,
    )

    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=6)

    # Save directory
    save_dir_pred = DATA_PATH + 'predictions/' + model_name + save_pred_ext
    save_dir_pred_pal = DATA_PATH + 'predictions_palette/' + model_name + save_pred_ext
    save_dir_pred_pal_cat = DATA_PATH + 'predictions_palette_cat/' + model_name + save_pred_ext
    if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)
    if not os.path.exists(save_dir_pred_pal): os.makedirs(save_dir_pred_pal)
    if not os.path.exists(save_dir_pred_pal_cat): os.makedirs(save_dir_pred_pal_cat)

    # Create a dictionary to store metrics
    metric = {} # Nested metric format: metric[image_name][label] = [precision, recall, dice, iou]

    # fig, ax = plt.subplots(5,2, figsize=(10,15))
    iter_test_dataloader = iter(test_dataloader)

    palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    stp, stn, sfp, sfn = 0, 0, 0, 0

    for i in range(len(list_IDs_test)):

        tp, tn, fp, fn = 0, 0, 0, 0

        name = os.path.splitext(list_IDs_test[i])[0] # remove extension

        metric[name] = {} # Creating nested dictionary

        # Image-wise mean of metrics
        i_mp, i_mr, i_mdice, i_miou = [], [], [], []

        image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

        # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])

        pr_mask = model.predict(image.to(DEVICE)) # Move image tensor to gpu

        # Convert from onehot
        # gt_mask = torch.argmax(gt_mask_, dim=1)
        if TO_CATEGORICAL:
            pr_mask = torch.argmax(pr_mask, dim=1)

        # pr_mask = torch.argmax(pr_mask, dim=1)

        # Move to CPU and convert to numpy
        gt_mask = gt_mask.squeeze().cpu().numpy()
        gt_mask = np.asarray(gt_mask, dtype=np.int64) # convert to integer
        pred = pr_mask.squeeze().cpu().numpy()

        # Save raw prediction
        if RAW_PREDICTION: raw_pred.append(pred)

        # Modify prediction based on threshold
        # pred = (pred >= threshold) * 1

        # Save prediction as png
        if save_pred:
            "Uncomment for non-palette"
            cv2.imwrite(os.path.join(save_dir_pred, list_IDs_test[i]), np.squeeze(pred).astype(np.uint8))

            "Uncomment for palette"
            # Palette original
            pal_gt_mask = np.squeeze(gt_mask).astype(np.uint8)
            pal_gt_mask = Image.fromarray(pal_gt_mask)
            pal_gt_mask = pal_gt_mask.convert("P")
            pal_gt_mask.putpalette(np.array(palette, dtype=np.uint8))

            # Palette prediction
            pal_pred = np.squeeze(pred).astype(np.uint8)
            pal_pred = Image.fromarray(pal_pred)
            pal_pred = pal_pred.convert("P")
            pal_pred.putpalette(np.array(palette, dtype=np.uint8))

            pal_pred.save(os.path.join(save_dir_pred_pal, list_IDs_test[i])) # store

            # Concatenate gt and pred side by side
            concat_pals = Image.new("RGB", (pal_gt_mask.width+pal_gt_mask.width, pal_gt_mask.height), "white")
            concat_pals.paste(pal_gt_mask, (0, 0))
            concat_pals.paste(pal_pred, (pal_gt_mask.width, 0))

            concat_pals.save(os.path.join(save_dir_pred_pal_cat, list_IDs_test[i])) # store

        # Find labels in gt and prediction
        lbl_gt = set(np.unique(gt_mask))
        lbl_gt.remove(0) # remove 0. It is background
        lbl_pred = set(np.unique(pred))
        lbl_pred.remove(0) # remove 0. It is background

        # All labels
        all_lbls = lbl_gt.union(lbl_pred)

        # Find labels that are not common in both gt and prediction. For such cases. IoU = 0
        diff1 = lbl_gt - lbl_pred
        diff2 = lbl_pred - lbl_gt
        diffs = diff1.union(diff2) # labels that do not exist in either gt or prediction

        # Labels that are in the gt but not in prediction are fn
        if len(diff1) > 0:
            for d1 in diff1:
                fn_ = len(np.argwhere(gt_mask == d1))
                fn += fn_
                sfn += fn

        # Labels that are in the prediction but not in gt are fp
        if len(diff2) > 0:
            for d2 in diff2:
                fp_ = len(np.argwhere(pred == d2))
                fp += fp_
                sfp += fp

        # Set IoU == 0 for such labels
        if not len(diffs) == 0:
            for diff in diffs:
                p, r, dice, iou = 0, 0, 0, 0
                metric[name][str(diff)] = [p, r, dice, iou]
                print("%d %s: label: %s; Precision: %3.2f; Recall: %3.2f; Dice: %3.2f; IoU: %3.2f"%(i+1, name, diff, p, r, dice, iou))

        # Find labels that are common in both gt and prediction.
        cmns = lbl_gt.intersection(lbl_pred)

        # Iterate over common labels
        for cmn in cmns:
            gt_idx = np.where(gt_mask == cmn)
            pred_idx = np.where(pred == cmn)

            # Convert to [(x1,y1), (x2,y2), ...]
            gt_lidx, pred_lidx = [], [] # List index

            for i in range(len(gt_idx[0])):
                gt_lidx.append((gt_idx[0][i], gt_idx[1][i]))

            for i in range(len(pred_idx[0])):
                pred_lidx.append((pred_idx[0][i], pred_idx[1][i]))

            # Calculate metrics
            gt_tidx = tuple(gt_lidx) # convert to tuple
            pred_tidx = tuple(pred_lidx) # convert to tuple
            tp_cord = set(gt_tidx).intersection(pred_tidx) # set operation
            fp_cord = set(pred_tidx).difference(gt_tidx) # set operation
            fn_cord = set(gt_tidx).difference(pred_tidx) # set operation

            tp += len(tp_cord)
            fp += len(fp_cord)
            fn += len(fn_cord)

            stp += tp
            sfp += fp
            sfn += fn

            p = (tp/(tp + fp + ep)) * 100
            r = (tp/(tp + fn + ep)) * 100
            dice = (2 * tp / (2 * tp + fp + fn + ep)) * 100
            iou = (tp/(tp + fp + fn + ep)) * 100

            print("%d %s: label: %s; Precision: %3.2f; Recall: %3.2f; Dice: %3.2f; IoU: %3.2f"%(i+1, name, cmn, p, r, dice, iou))

            metric[name][str(cmn)] = [p, r, dice, iou]

            # Keep appending metrics for all labels for the current image
            i_mp.append(p)
            i_mr.append(r)
            i_mdice.append(dice)
            i_miou.append(iou)


    # create json object from dictionary
    import json
    json_write = json.dumps(metric)
    f = open(os.path.join(save_dir_pred, "metric.json"), "w")
    f.write(json_write)
    f.close()

    # Data-based evalutation
    siou = (stp/(stp + sfp + sfn + ep))*100
    sprecision = (stp/(stp + sfp + ep))*100
    srecall = (stp/(stp + sfn + ep))*100
    sdice = (2 * stp / (2 * stp + sfp + sfn))*100

    print('siou:', siou)
    print('sprecision:', sprecision)
    print('srecall:', srecall)
    print('sdice:', sdice)

    # Save data-based result in a text file
    with open(os.path.join(save_dir_pred, 'result.txt'), 'w') as f:
        print(f'iou = {siou}', file=f)
        print(f'precision = {sprecision}', file=f)
        print(f'recall = {srecall}', file=f)
        print(f'dice = {sdice}', file=f)
        print(f'model name = {model_name}', file=f)

    return save_dir_pred_pal_cat, sdice
