import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

from src.patchcore.prepare_private_submission import save_image_segmentation_submission
import src.patchcore.backbones as backbones
import src.patchcore.common
import src.patchcore.datasets
import src.patchcore.metrics as metrics
import src.patchcore.patchcore as patchcore
from src.patchcore.sampler import (
    ApproximateGreedyCoresetSamplerCosine,
    IdentitySampler,
    GreedyCoresetSampler,
    ApproximateGreedyCoresetSampler,
)
import src.patchcore.utils as utils
from src.patchcore.datasets import (
    MVTecDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MVTec2Dataset,
    MVTecDataset_,
    PrivateMVTec2Dataset,
)
from src.datasets.mvtec_outlier_ds import MvtecOutlierDataset
from torchvision.transforms import transforms as T
import pandas as pd
import itertools
import csv
from src.mvtec_ad_evaluation.evaluate_experiment import calculate_au_pro_au_roc
from src.patchcore.transforms_utils import (
    ResizeLongestSide,
    PadToSquareTensor,
    remove_padding,
)

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}

config = {}


def write_results_to_csv(all_results_csv_file, config, metrics):
    """
    Write the results of the PatchCore experiment into a CSV file.

    Args:
        results_path (str): Path to the results directory.
        all_results_csv_file (str): Name of the CSV file to store results.
        config (dict): Configuration used for the experiment.
        metrics (dict): Metrics results including mean and std over seeds.
    """
    file_exists = os.path.isfile(all_results_csv_file)

    # Define the header for the CSV file
    header = [
        "category",
        "dataset",
        "synth_anomalies_path",
        "backbone",
        "layers",
        "anomalies_lower_threshold",
        "anomalies_upper_threshold",
        "alpha",
        "anomaly_scorer_num_nn",
        "score_type",
        "metric_type",
        "num_seeds",
        "image_auc_mean",
        "image_auc_std",
        "pixel_auc_mean",
        "pixel_auc_std",
        "anomaly_pixel_auc_mean",
        "anomaly_pixel_auc_std",
        "aupro_0.05_mean",
        "aupro_0.05_std",
        "aupro_0.3_mean",
        "aupro_0.3_std",
        "optimal_threshold_mean",
        "optimal_threshold_std",
    ]

    # Open the CSV file in append mode
    with open(all_results_csv_file, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Prepare the row data
        row = {
            "category": config.get("category"),
            "dataset": config.get("dataset"),
            "synth_anomalies_path": config.get("synth_anomalies_path"),
            "backbone": config.get("backbone"),
            "layers": config.get("layers"),
            "anomalies_lower_threshold": config.get("anomalies_lower_threshold"),
            "anomalies_upper_threshold": config.get("anomalies_upper_threshold"),
            "alpha": config.get("alpha"),
            "anomaly_scorer_num_nn": config.get("anomaly_scorer_num_nn"),
            "score_type": config.get("score_type"),
            "metric_type": config.get("metric_type"),
            "num_seeds": config.get("num_seeds"),
            "image_auc_mean": metrics.get("image_auc_mean"),
            "image_auc_std": metrics.get("image_auc_std"),
            "pixel_auc_mean": metrics.get("pixel_auc_mean"),
            "pixel_auc_std": metrics.get("pixel_auc_std"),
            "anomaly_pixel_auc_mean": metrics.get("anomaly_pixel_auc_mean"),
            "anomaly_pixel_auc_std": metrics.get("anomaly_pixel_auc_std"),
            "aupro_0.05_mean": metrics.get("aupro_0.05_mean"),
            "aupro_0.05_std": metrics.get("aupro_0.05_std"),
            "aupro_0.3_mean": metrics.get("aupro_0.3_mean"),
            "optimal_threshold_mean": metrics.get("optimal_threshold_mean"),
            "optimal_threshold_std": metrics.get("optimal_threshold_std"),
        }

        # Write the row to the CSV file
        writer.writerow(row)

    print(f"Results written to {all_results_csv_file}")


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[], multiple=True, show_default=True)
@click.option("--num_seeds", type=int, default=1, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--save_private_submission", is_flag=True)
@click.option("--all_results_csv_file", type=str, default="patchcore_results.csv")
@click.option("--submission_dir", type=str, default=None)
@click.option("--num_stds", type=float, default=3.0)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    num_seeds,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    all_results_csv_file,
    save_private_submission: bool,
    submission_dir: str,
    num_stds: float
):
    methods = {key: item for (key, item) in methods}

    if save_private_submission and submission_dir is None:
        raise ValueError("If you want tot save private submission, provide a path")

    def log(masks, ndists, andists, neighbors):
        df = pd.DataFrame()

        df["labels_gt"] = masks
        df["ndists"] = ndists
        df["andists"] = andists

        (
            normal_imgs,
            normal_patch_idx,
            not_normal_imgs,
            not_normal_patch_idx,
            ndist,
            notndist,
            names,
        ) = zip(*neighbors)

        neighbors_df = pd.DataFrame()
        neighbors_df["normal_imgs"] = normal_imgs
        neighbors_df["normal_patch_idx"] = normal_patch_idx
        neighbors_df["not_normal_imgs"] = not_normal_imgs
        neighbors_df["not_normal_patch_idx"] = not_normal_patch_idx
        neighbors_df["test_img_path"] = names
        neighbors_df["test_img_patch_id"] = neighbors_df.index % (
            len(neighbors_df) // len(set(names))
        )
        neighbors_df["ndist"] = ndist
        neighbors_df["not_ndist"] = notndist

        save_path = os.path.join(run_save_path, "patchcore_anomaly_scores.csv")
        neighbors_save_path = os.path.join(run_save_path, "patchcore_neighbors.csv")
        neighbors_df.to_csv(neighbors_save_path, index=False)
        df.to_csv(save_path, index=False)

    metrics_dict = {
        "image_aucs": [],
        "pixel_aucs": [],
        "anomaly_pixel_aucs": [],
        "mvtec2_aucs": [],
        "aupro_0.05": [],
        "optimal_threshold": [],
        "aupro_0.3": [],
    }

    for seed in range(42, 42+num_seeds):

        run_save_path = utils.create_storage_folder(
            results_path, log_project, log_group, mode="iterate"
        )

        list_of_dataloaders = methods["get_dataloaders"](seed)

        device = utils.set_torch_device(gpu)
        # Device context here is specifically set and used later
        # because there was GPU memory-bleeding which I could only fix with
        # context managers.
        device_context = (
            torch.cuda.device("cuda:{}".format(device.index))
            if "cuda" in device.type.lower()
            else contextlib.suppress()
        )

        result_collect = []

        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            LOGGER.info(
                "Evaluating dataset [{}] ({}/{})...".format(
                    dataloaders["training"].name,
                    dataloader_count + 1,
                    len(list_of_dataloaders),
                )
            )

            utils.fix_seeds(seed, device)

            dataset_name = dataloaders["training"].name

            with device_context:
                torch.cuda.empty_cache()
                imagesize = dataloaders["training"].dataset.imagesize
                sampler = methods["get_sampler"](
                    device,
                )
                PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
                if len(PatchCore_list) > 1:
                    LOGGER.info(
                        "Utilizing PatchCore Ensemble (N={}).".format(
                            len(PatchCore_list)
                        )
                    )
                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()
                    if PatchCore.backbone.seed is not None:
                        utils.fix_seeds(PatchCore.backbone.seed, device)
                    LOGGER.info(
                        "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                    )
                    torch.cuda.empty_cache()
                    PatchCore.fit(
                        dataloaders["training"], dataloaders["synth_anomalies"]
                    )
                    np.save(
                        os.path.join(run_save_path, "normal_scores.npy"),
                        PatchCore.normal_scores,
                    )
                    np.save(
                        os.path.join(run_save_path, "anomaly_scores.npy"),
                        PatchCore.anomaly_scores,
                    )
                    if dataloaders.get("validation"):
                        threshold = PatchCore.validate(dataloaders["validation"], num_stds)
                        LOGGER.info(
                            f"Determined threshold on validation set: {threshold}"
                        )

                torch.cuda.empty_cache()
                aggregator = {"scores": [], "segmentations": []}
                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()
                    LOGGER.info(
                        "Embedding test data with models ({}/{})".format(
                            i + 1, len(PatchCore_list)
                        )
                    )
                    print(
                        f"Size of test dataset: {len(dataloaders['testing'].dataset)}"
                    )
                    (
                        scores,
                        segmentations,
                        labels_gt,
                        masks_gt,
                        ndists,
                        andists,
                        neighbors,
                    ) = PatchCore.predict(dataloaders["testing"])
                    if save_private_submission:
                        private_preds = PatchCore.predict(
                            dataloaders["private_testing"]
                        )
                        priv_scores = private_preds[0]
                        priv_segmentations = private_preds[1]
                    else:
                        priv_scores = None
                        priv_segmentations = None

                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)
                    aggregator["ndists"] = ndists
                    aggregator["andists"] = andists
                    aggregator["neighbors"] = neighbors

                scores = np.array(aggregator["scores"])
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores)
                scores = np.mean(scores, axis=0)

                segmentations = np.array(aggregator["segmentations"])
                orig_shape = segmentations.shape

                # remove padding

                unpadded_segmentations = []
                unpadded_masks = []
                masks_gt = np.array(masks_gt)
                padding_metadata = dataloaders["testing"].dataset.padding_metadata

                for i in range(segmentations.shape[1]):

                    unpadded_tensor = remove_padding(
                        torch.from_numpy(segmentations[:, i, :, :]), padding_metadata
                    )
                    unpadded_segmentations.append(unpadded_tensor.numpy())
                    # remove from masks_gt as well
                    unpadded_mask_tensor = remove_padding(
                        torch.from_numpy(masks_gt[i]), padding_metadata
                    )
                    unpadded_masks.append(unpadded_mask_tensor.numpy())
                segmentations = np.array(unpadded_segmentations)
                new_shape = segmentations.shape
                segmentations = segmentations.reshape(
                    (orig_shape[0], orig_shape[1], new_shape[2], new_shape[3])
                )
                masks_gt = np.array(unpadded_masks)
                print(f"Maximum ground truth pixel: {masks_gt[masks_gt > 0.0].max()}")

                min_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .min(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                max_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .max(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                np.save(
                    f"{dataloaders['training'].dataset.classnames_to_use[0]}_gt_public.npy",
                    masks_gt,
                )
                np.save(
                    f"{dataloaders['training'].dataset.classnames_to_use[0]}_segmentations_public.npy",
                    segmentations,
                )
                optimal_threshold = metrics.compute_pixelwise_best_f1_threshold(
                    anomaly_segmentations=segmentations, ground_truth_masks=masks_gt
                )
                segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                segmentations = np.mean(segmentations, axis=0)

                anomaly_labels = [
                    x[1] != "good"
                    for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                test_images = [
                    x[0] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                # (Optional) Plot example images.
                if save_segmentation_images:
                    image_paths = [
                        x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                    ]

                    def image_transform(image):
                        in_std = np.array(
                            dataloaders["testing"].dataset.transform_std
                        ).reshape(-1, 1, 1)
                        in_mean = np.array(
                            dataloaders["testing"].dataset.transform_mean
                        ).reshape(-1, 1, 1)
                        image = dataloaders["testing"].dataset.transform_img(image)
                        return np.clip(
                            (image.numpy() * in_std + in_mean) * 255, 0, 255
                        ).astype(np.uint8)

                    def mask_transform(mask):
                        return (
                            dataloaders["testing"].dataset.transform_mask(mask).numpy()
                        )

                    image_save_path = os.path.join(
                        run_save_path, "segmentation_images", dataset_name
                    )
                    os.makedirs(image_save_path, exist_ok=True)
                    utils.plot_segmentation_images(
                        image_save_path,
                        image_paths,
                        segmentations,
                        scores,
                        mask_paths,
                        image_transform=image_transform,
                        mask_transform=mask_transform,
                    )

                anomaly_maps_path = os.path.join(run_save_path, "anomaly_maps")
                os.makedirs(anomaly_maps_path, exist_ok=True)
                _, _, _, _, _, _, names = zip(*aggregator["neighbors"])
                test_image_names = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                assert len(test_image_names) == len(segmentations)
                assert len(test_image_names) == len(masks_gt)
                anomaly_maps = utils.save_anomaly_maps(
                    segmentations, anomaly_maps_path, test_image_names
                )
                gt_maps_path = os.path.join(run_save_path, "gt_maps")
                os.makedirs(gt_maps_path, exist_ok=True)
                gt_maps = utils.save_gt_maps(test_image_names, gt_maps_path, masks_gt)
                aupro, mvtec2_auroc, _, _ = calculate_au_pro_au_roc(
                    gt_maps, anomaly_maps, 0.05
                )

                aupro_30, _, _, _ = calculate_au_pro_au_roc(gt_maps, anomaly_maps, 0.3)

                masks_gt_ = np.stack(masks_gt).reshape(-1)
                ndists = np.stack(ndists).reshape(-1)
                andists = np.stack(andists).reshape(-1)
                ndists = ndists.reshape(
                    len(segmentations), orig_shape[2], orig_shape[3]
                )
                andists = andists.reshape(
                    len(segmentations), orig_shape[2], orig_shape[3]
                )
                ndists_ = []
                andists_ = []
                for ndist, andist in zip(ndists, andists):
                    ndist_tens = torch.from_numpy(ndist)
                    ndists_.append(remove_padding(ndist_tens, padding_metadata))
                    andist_tens = torch.from_numpy(andist)
                    andists_.append(remove_padding(andist_tens, padding_metadata))

                ndists = np.array(ndists_).reshape(-1)
                andists = np.array(andists_).reshape(-1)

                log(masks_gt_, ndists, andists, neighbors)

                LOGGER.info("Computing evaluation metrics.")
                auroc = metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["auroc"]

                # Compute PRO score & PW Auroc for all images
                pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt
                )

                full_pixel_auroc = pixel_scores["auroc"]

                # Compute PRO score & PW Auroc only images with anomalies
                sel_idxs = []
                for i in range(len(masks_gt)):
                    if np.sum(masks_gt[i]) > 0:
                        sel_idxs.append(i)
                pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores["auroc"]

                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "instance_auroc": auroc,
                        "full_pixel_auroc": full_pixel_auroc,
                        "anomaly_pixel_auroc": anomaly_pixel_auroc,
                        "mvtec2_auroc": mvtec2_auroc,
                        "aupro_0.05": aupro,
                        "aupro_0.3": aupro_30,
                    }
                )
                metrics_dict["image_aucs"].append(auroc)
                metrics_dict["pixel_aucs"].append(full_pixel_auroc)
                metrics_dict["anomaly_pixel_aucs"].append(anomaly_pixel_auroc)
                metrics_dict["mvtec2_aucs"].append(mvtec2_auroc)
                metrics_dict["aupro_0.05"].append(aupro)
                metrics_dict["optimal_threshold"].append(optimal_threshold)
                metrics_dict["aupro_0.3"].append(aupro_30)

                for key, item in result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:3.3f}".format(key, item))

                if save_private_submission:
                    unpadded_segmentations = []
                    priv_segmentations = np.array(priv_segmentations)

                    for i in range(priv_segmentations.shape[0]):
                        unpadded_tensor = remove_padding(
                            torch.from_numpy(priv_segmentations[i, :, :]),
                            dataloaders["testing"].dataset.padding_metadata,
                        )
                        unpadded_segmentations.append(unpadded_tensor.numpy())
                    priv_segmentations = np.array(unpadded_segmentations)
                    new_shape = segmentations.shape
                    # priv_segmentations = priv_segmentations.reshape(
                    #     (orig_shape[0], orig_shape[1], new_shape[2], new_shape[3])
                    # )
                    priv_image_names = [
                        x[1]
                        for x in dataloaders["private_testing"].dataset.data_to_iterate
                    ]
                    priv_segmentations_thresholded = (
                        priv_segmentations >= threshold
                    )
                    min_scores = priv_segmentations.min()
                    max_scores = priv_segmentations.max()
                    priv_segmentations = (priv_segmentations - min_scores) / (
                        max_scores - min_scores
                    )
                    priv_segmentations_thresholded = (
                        priv_segmentations_thresholded.reshape(priv_segmentations.shape)
                    )
                    save_image_segmentation_submission(
                        submission_dir,
                        classname=dataloaders["training"].dataset.classnames_to_use[0],
                        image_paths=priv_image_names,
                        segmentations=priv_segmentations,
                        segmentations_thresholded=priv_segmentations_thresholded,
                    )

                # (Optional) Store PatchCore model for later re-use.
                # SAVE all patchcores only if mean_threshold is passed?
                if save_patchcore_model:
                    patchcore_save_path = os.path.join(
                        run_save_path, "models", dataset_name
                    )
                    os.makedirs(patchcore_save_path, exist_ok=True)
                    for i, PatchCore in enumerate(PatchCore_list):
                        prepend = (
                            "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                            if len(PatchCore_list) > 1
                            else ""
                        )
                        PatchCore.save_to_path(patchcore_save_path, prepend)

            LOGGER.info("\n\n-----\n")

        # Store all results and mean scores to a csv-file.
        result_metric_names = list(result_collect[-1].keys())[1:]
        result_dataset_names = [results["dataset_name"] for results in result_collect]
        result_scores = [list(results.values())[1:] for results in result_collect]
        utils.compute_and_store_final_results(
            run_save_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_dataset_names,
        )
    agg_metrics = {
        "image_auc_mean": np.mean(metrics_dict["image_aucs"]),
        "image_auc_std": np.std(metrics_dict["image_aucs"]),
        "pixel_auc_mean": np.mean(metrics_dict["pixel_aucs"]),
        "pixel_auc_std": np.std(metrics_dict["pixel_aucs"]),
        "anomaly_pixel_auc_mean": np.mean(metrics_dict["anomaly_pixel_aucs"]),
        "anomaly_pixel_auc_std": np.std(metrics_dict["anomaly_pixel_aucs"]),
        "mvtec2_auc_mean": np.mean(metrics_dict["mvtec2_aucs"]),
        "mvtec2_auc_std": np.std(metrics_dict["mvtec2_aucs"]),
        "aupro_0.05_mean": np.mean(metrics_dict["aupro_0.05"]),
        "aupro_0.05_std": np.std(metrics_dict["aupro_0.05"]),
        "aupro_0.3_mean": np.mean(metrics_dict["aupro_0.3"]),
        "aupro_0.3_std": np.std(metrics_dict["aupro_0.3"]),
        "optimal_threshold_mean": np.mean(metrics_dict["optimal_threshold"]),
        "optimal_threshold_std": np.std(metrics_dict["optimal_threshold"]),
    }
    config["num_seeds"] = num_seeds

    write_results_to_csv(all_results_csv_file, config, agg_metrics)


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--metric_type", type=click.Choice(["L2", "IP"]), default="L2")
@click.option("--score", type=click.Choice(["mean", "last", "median"]), default="mean")
@click.option("--anomalies_cutoff", type=float, default=0.75)
@click.option("--anomalies_lower_threshold", "-alt", type=float, default=85.0)
@click.option("--anomalies_upper_threshold", "-aut", type=float, default=100.0)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
@click.option("--alpha", type=float, default=0.0)
@click.option(
    "--anomaly_score_fn", type=click.Choice(["ratio", "diff"]), default="ratio"
)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
    alpha,
    metric_type,
    score,
    anomalies_cutoff,
    anomalies_lower_threshold,
    anomalies_upper_threshold,
    anomaly_score_fn,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    config["anomaly_scorer_num_nn"] = anomaly_scorer_num_nn
    config["backbone"] = str.join(",", backbone_names)
    config["layers"] = str.join(",", layers_to_extract_from)
    config["alpha"] = alpha
    config["score_type"] = score
    config["metric_type"] = metric_type
    config["anomalies_lower_threshold"] = anomalies_lower_threshold
    config["anomalies_upper_threshold"] = anomalies_upper_threshold
    config["anomaly_score_fn"] = anomaly_score_fn

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = src.patchcore.common.FaissNN(
                faiss_on_gpu, faiss_num_workers, metric_type=metric_type
            )
            not_normal_nn_method = src.patchcore.common.FaissNN(
                faiss_on_gpu, faiss_num_workers, metric_type=metric_type
            )

            patchcore_instance = patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_score_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                not_normal_nn_method=not_normal_nn_method,
                alpha=alpha,
                anomalous_featuresampler=ApproximateGreedyCoresetSampler(
                    0.85, device=device
                ),
                score=score,
                anomalies_cutoff=anomalies_cutoff,
                anomalies_bottom_threshold=anomalies_lower_threshold,
                anomalies_top_threshold=anomalies_upper_threshold,
                anomaly_score_fn=anomaly_score_fn,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return IdentitySampler()
        elif name == "greedy_coreset":
            return GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return ApproximateGreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset_cosine":
            return ApproximateGreedyCoresetSamplerCosine(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
@click.option(
    "--dataset-type",
    "--dataset_type",
    type=click.Choice(["mvtec", "mvtec2"]),
    default="mvtec",
)
@click.option(
    "--test_anomaly_types",
    "-tat",
    multiple=True,
    type=str,
    default=["hole", "cut", "crack", "print"],
)
@click.option("--train_anomaly_types", "-trat", multiple=True, type=str, default=[])
@click.option("--mini", is_flag=True)
@click.option("--apply_augmentations", is_flag=True)
@click.option("--apply_padding", is_flag=True)
@click.option(
    "--synth_anomalies_path",
    type=click.Path(exists=True, file_okay=False),
    default=None,
)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
    dataset_type,
    synth_anomalies_path,
    train_anomaly_types,
    test_anomaly_types,
    mini,
    apply_padding,
    apply_augmentations,
):
    # dataset_info = _DATASETS[name]
    # dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    ds_class = MVTecDataset if dataset_type == "mvtec" else MVTec2Dataset
    trainsplit = src.patchcore.datasets.DatasetSplit.TRAIN

    config["dataset"] = dataset_type
    config["synth_anomalies_path"] = synth_anomalies_path
    config["category"] = subdatasets[0]

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = ds_class(
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=1.0,
                imagesize=imagesize,
                split=trainsplit,
                seed=seed,
                augment=augment,
                mini=mini,
                do_padding=apply_padding,
            )
            print(
                f"Training dataset size for {subdataset}: {len(train_dataset)} samples. Mini: {mini}"
            )

            # test_dataset = ds_class(
            #     data_path,
            #     classname=subdataset,
            #     resize=resize,
            #     imagesize=imagesize,
            #     split=src.patchcore.datasets.DatasetSplit.TEST,
            #     seed=seed,
            # )
            if train_anomaly_types and test_anomaly_types:
                test_dataset = MVTecDataset_(
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=src.patchcore.datasets.DatasetSplit.TEST,
                    seed=seed,
                    anomaly_types=test_anomaly_types,
                    mini=mini,
                )
            elif dataset_type == "mvtec":
                test_dataset = MVTecDataset(
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    do_padding=apply_padding,
                    split=src.patchcore.datasets.DatasetSplit.TEST,
                    seed=seed,
                )
            else:
                test_dataset = MVTec2Dataset(
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    do_padding=apply_padding,
                    split=src.patchcore.datasets.DatasetSplit.TEST,
                    seed=seed,
                )

            # test_dataset = MvtecOutlierDataset(synth_anomalies_path, category=subdataset,
            #                                               transform=T.Compose([
            #                                                     T.Resize((resize, resize)),
            #                                                     T.CenterCrop((imagesize, imagesize)),
            #                                                     T.ToTensor(),
            #                                                     T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            #                                                     ]),
            #                                               mask_transform=T.Compose([
            #                                                     T.Resize((resize, resize)),
            #                                                     T.CenterCrop((imagesize, imagesize)),
            #                                                     T.ToTensor()])
            #                                               )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            if train_anomaly_types and test_anomaly_types:
                print("Using test dataset as outliers")
                synth_anomalies_dataset = MVTecDataset_(
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=src.patchcore.datasets.DatasetSplit.TEST,
                    seed=seed,
                    anomaly_types=train_anomaly_types,
                    mini=mini,
                )
            else:
                if apply_padding:
                    img_transform = T.Compose(
                        [
                            ResizeLongestSide(imagesize),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                            PadToSquareTensor(target_size=imagesize),
                        ]
                    )
                    mask_transform = T.Compose(
                        [
                            ResizeLongestSide(imagesize),
                            T.ToTensor(),
                            PadToSquareTensor(target_size=imagesize),
                        ]
                    )
                else:
                    img_transform = T.Compose(
                        [
                            T.Resize((resize, resize)),
                            T.CenterCrop((imagesize, imagesize)),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ]
                    )
                    mask_transform = T.Compose(
                        [
                            T.Resize((resize, resize)),
                            T.CenterCrop((imagesize, imagesize)),
                            T.ToTensor(),
                        ]
                    )
                synth_anomalies_dataset = MvtecOutlierDataset(
                    synth_anomalies_path,
                    category=subdataset,
                    transform=img_transform,
                    mask_transform=mask_transform,
                    apply_augmentations=apply_augmentations,
                )
            print(
                f"Synthetic anomalies dataset size for {subdataset}: {len(synth_anomalies_dataset)} samples."
            )

            synth_anomalies_dataloader = torch.utils.data.DataLoader(
                synth_anomalies_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset
            
            if name == 'mvtec2':
                private_ds = PrivateMVTec2Dataset(
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    seed=seed,
                    do_padding=apply_padding,
                )

                private_ds_loader = torch.utils.data.DataLoader(
                    private_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                private_ds_loader = None

            if dataset_type == "mvtec2":
                val_dataset = ds_class(
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=src.patchcore.datasets.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "synth_anomalies": synth_anomalies_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
                "private_testing": private_ds_loader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
