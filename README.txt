Description of "rvae_config.json" (how to customize each experiment):

* model_type: several kind of models are available.
    (1) model_types.BASELINE;
    (2) model_types.LOW, focused on maximizing accuracy on low-popular items; RVAE_L
    (3) model_types.MED, focused on maximizing accuracy on mid-popular items;
    (4) model_types.HIGH, focused on maximizing accuracy on high-popular items;
    (5) model_types.WEIGHTING, RVAE_W;
    (6) model_types.OVERSAMPLING, RVAE_S.

* dataset_name: several datasets are available. You can find more details in "datasets_info.json" file.

* copy_pasting_data: This is a boolean flag taking values {True, False}. When it is set to True, the experiment directory
                    is copy-pasted in the main folder (i.e.: low_YMD_HM -> ../low)

* best_model_k_metric: This parameter takes three possible values {1, 5, 10}, it is used to choose the metric@k to take
                        into account when selecting the best model during the validation phase.

