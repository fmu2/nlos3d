import yaml


RSD_DEFAULTS = {
  "meas": {
    "ds": 1,
    "scale": 1,
    "background": 0,
    "target_noise": 0,
  },

  "rsd": {
    "wall_size": 2,
    "zmin": 0,
    "zmax": 2,
    "n_cycles": 4,
    "ratio": 0.1,

    "actv": "linear",
    "norm": "none",
    "affine": False,
  },
}


TRAIN_NR_DEFAULTS = {
  "dataset": {
    "ds": 1,
    "scale": 1,
    "background": 0,
    "target_noise": 0,
  },

  "model": {
    "encoder": {},

    "decoder": {
      "transformer": {
        "wall_size": 2,
        "zmin": -1,
        "zmax": 1,
      },
    },
  },

  "opt": {
    "batch_size": 4,
    "n_itrs": 1e5,

    "optim_type": "adam",
    "lr": 1e-4,
    "weight_decay": 0,
    "beta1": 0.9,
    "beta2": 0.999,

    "clip_grad_norm": 1,

    "sched_type": "step",
    "milestones": [-1],
    "gamma": 0.1,
  },

  "train": {
    "n_views": 4,
    "include_orthogonal": True,
    "in_scale": None,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "in_scale": None,
  },
}


TRAIN_SUP_DEFAULTS = {
  "dataset": {
    "ds": 1,
    "scale": 1,
    "background": 0,
    "target_noise": 0,
  },

  "camera": {
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
    },
  },

  "model": {
    "encoder": {},

    "renderer": {
      "embedder": {
        "embed_p": True,
        "embed_d": True,
        "embed_z": False,

        "p": {
          "in_dim": 3,
          "include_input": True,
          "n_freqs": 6,
          "log_sampling": True,
          "freq_scale": 1,
        },
        
        "d": {
          "in_dim": 2,
          "include_input": False,
          "n_freqs": 4,
          "log_sampling": True,
          "freq_scale": 1,
        },
        
        "z": {},
      },
      
      "common": {
        "bb_ctr": [0, 0, 0],
        "bb_size": [2, 2, 2],
        "inf": 10,
        "p_polar": False,
        "d_polar": True,
        "z_norm": False,
        "sigma_transform": "relu",
        "color_transform": "relu",
      },

      "steady_state": {
        "bin_len": 0.02,
        "n_bins": 128,
      },
    },
  },

  "opt": {
    "batch_size": 4,
    "n_itrs": 1e5,

    "optim_type": "adam",
    "lr": 1e-4,
    "weight_decay": 0,
    "beta1": 0.9,
    "beta2": 0.999,

    "clip_grad_norm": 1,

    "sched_type": "step",
    "milestones": [-1],
    "gamma": 0.1,

    "mse": 1.0,
    "beta": 0.0,
    "tv": 0.0,
  },

  "train": {
    "n_views": 4,
    "include_orthogonal": True,

    "in_scale": 1,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "chunk_size": 4096,

    "in_scale": 1,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },
}


TRAIN_UNSUP_DEFAULTS = {
  "dataset": {
    "ds": 1,
    "scale": 1,
    "background": 0,
    "target_noise": 0,
  },

  "camera": {
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
    },
  },

  "wall": {
    "size": 2,
    "uv_size": 64,
    "foreshortening": False,
  },

  "model": {
    "encoder": {},

    "renderer": {
      "embedder": {
        "embed_p": True,
        "embed_d": True,
        "embed_z": False,

        "p": {
          "in_dim": 3,
          "include_input": True,
          "n_freqs": 6,
          "log_sampling": True,
          "freq_scale": 1,
        },
        
        "d": {
          "in_dim": 2,
          "include_input": False,
          "n_freqs": 4,
          "log_sampling": True,
          "freq_scale": 1,
        },
        
        "z": {},
      },
      
      "common": {
        "bb_ctr": [0, 0, 0],
        "bb_size": [2, 2, 2],
        "inf": 10,
        "p_polar": False,
        "d_polar": True,
        "z_norm": False,
        "sigma_transform": "relu",
        "color_transform": "relu",
      },

      "transient": {
        "light": [0, 0, -1],
      },

      "steady_state": {
        "bin_len": 0.02,
        "n_bins": 128,
      },
    },
  },

  "opt": {
    "batch_size": 4,
    "n_itrs": 1e5,

    "optim_type": "adam",
    "lr": 1e-4,
    "weight_decay": 0,
    "beta1": 0.9,
    "beta2": 0.999,

    "clip_grad_norm": 1,

    "sched_type": "step",
    "milestones": [-1],
    "gamma": 0.1,

    "poisson": 1.0,
    "beta": 0.0,
    "tv": 0.0,
  },

  "train": {
    "in_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "chunk_size": 4096,

    "in_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },
}


RSD_PER_SCENE_DEFAULTS = {
  "dataset": {
    "ds": 1,
    "scale": 1,
    "background": 0,
  },

  "rsd": {
    "wall_size": 2,
    "zmin": 0,
    "zmax": 2,
    "n_cycles": 4,
    "ratio": 0.1,

    "actv": "linear",
    "norm": "none",
    "affine": False,
  },
}


TRAIN_SUP_PER_SCENE_DEFAULTS = {
  "target": {},

  "camera": {
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
    },
  },

  "model": {
    "embedder": {
      "embed_p": True,
      "embed_d": True,
      "embed_z": False,

      "p": {
        "in_dim": 3,
        "include_input": True,
        "n_freqs": 6,
        "log_sampling": True,
        "freq_scale": 1,
      },
      
      "d": {
        "in_dim": 2,
        "include_input": False,
        "n_freqs": 4,
        "log_sampling": True,
        "freq_scale": 1,
      },

      "z": {
        "in_dim": None,
      },
    },
    
    "field": {
      "type": "nerf",

      "nerf": {
        "hid_dim": 128,
        "skips": [5],
        "n_sigma_layers": 8,
        "n_color_layers": 1,
        "actv": "relu",
      },
    },
    
    "common": {
      "bb_ctr": [0, 0, 0],
      "bb_size": [2, 2, 2],
      "inf": 10,
      "p_polar": False,
      "d_polar": True,
      "z_norm": False,
      "sigma_transform": "relu",
      "color_transform": "relu",
    },

    "steady_state": {
      "bin_len": 0.02,
      "n_bins": 128,
    },
  },

  "opt": {
    "n_iters": 1e5,
    
    "optim_type": "adam",
    "lr": 1e-4,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,

    "clip_grad_norm": 1.0,

    "sched_type": "step",
    "milestones": [-1],
    "gamma": 0.1,

    "mse": 1,
    "beta": 0,
    "tv": 0,
  },

  "train": {
    "n_views": 4,
    "include_orthogonal": False,

    "n_steps": 2,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "chunk_size": 4096,

    "n_steps": 2,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },
}


TRAIN_UNSUP_PER_SCENE_DEFAULTS = {
  "meas": {
    "scale": 1,
    "background": 0,
  },

  "target": {},

  "camera": {
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
    },
  },

  "wall": {
    "size": 2,
    "uv_size": 64,
    "foreshortening": False,
  },

  "model": {
    "embedder": {
      "embed_p": True,
      "embed_d": True,
      "embed_z": False,

      "p": {
        "in_dim": 3,
        "include_input": True,
        "n_freqs": 6,
        "log_sampling": True,
        "freq_scale": 1,
      },
      
      "d": {
        "in_dim": 2,
        "include_input": False,
        "n_freqs": 4,
        "log_sampling": True,
        "freq_scale": 1,
      },

      "z": {
        "in_dim": None,
      },
    },
    
    "field": {
      "type": "nerf",

      "nerf": {
        "hid_dim": 128,
        "skips": [5],
        "n_sigma_layers": 8,
        "n_color_layers": 1,
        "actv": "relu",
      },
    },
    
    "common": {
      "bb_ctr": [0, 0, 0],
      "bb_size": [2, 2, 2],
      "inf": 10,
      "p_polar": False,
      "d_polar": True,
      "z_norm": False,
      "sigma_transform": "relu",
      "color_transform": "relu",
    },

    "transient": {
      "light": [0, 0, -1],
    },

    "steady_state": {
      "bin_len": 0.02,
      "n_bins": 128,
    },
  },

  "opt": {
    "n_iters": 1e5,
    
    "optim_type": "adam",
    "lr": 1e-4,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,

    "clip_grad_norm": 1.0,

    "sched_type": "step",
    "milestones": [-1],
    "gamma": 0.1,

    "poisson": 1,
    "beta": 0,
    "tv": 0,
  },

  "train": {
    "n_steps": 2,
    "t_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "chunk_size": 4096,

    "n_steps": 2,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },
}


TEST_NR_DEFAULTS = {
  "dataset": {
    "scale": 1,
    "ds": 1,
    "background": 0,
    "target_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "in_scale": 1,
  },
}

TEST_SUP_DEFAULTS = {
  "dataset": {
    "scale": 1,
    "ds": 1,
    "background": 0,
    "target_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "chunk_size": 4096,

    "in_scale": 1,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },
}

TEST_UNSUP_DEFAULTS = {
  "dataset": {
    "scale": 1,
    "ds": 1,
    "background": 0,
    "target_noise": 0,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "chunk_size": 4096,

    "in_scale": 1,
    "s_scale": 1,
    "sigma_noise": 0,
    "color_noise": 0,
  },
}


def _merge(src, dst):
  for k, v in src.items():
    if k in dst:
      if isinstance(v, dict):
        _merge(src[k], dst[k])
    else:
      dst[k] = v


def load_default_config(mode="train"):
  if mode == "rsd":
    config = RSD_DEFAULTS
  elif mode == "train_nr":
    config = TRAIN_NR_DEFAULTS
  elif mode == "train_sup":
    config = TRAIN_SUP_DEFAULTS
  elif mode == "train_unsup":
    config = TRAIN_UNSUP_DEFAULTS
  elif mode == "rsd_per_scene":
    config = RSD_PER_SCENE_DEFAULTS
  elif mode == "train_sup_per_scene":
    config = TRAIN_SUP_PER_SCENE_DEFAULTS
  elif mode == "train_unsup_per_scene":
    config = TRAIN_UNSUP_PER_SCENE_DEFAULTS
  elif mode == "test_nr":
    config = TEST_NR_DEFAULTS
  elif mode == "test_sup":
    config = TEST_SUP_DEFAULTS
  elif mode == "test_unsup":
    config = TEST_UNSUP_DEFAULTS
  else:
    raise ValueError("invalid default config mode: {:s}".format(mode))
  return config


def load_config(config_file, mode="train"):
  with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  if mode == "rsd":
    defaults = RSD_DEFAULTS
  elif mode == "train_nr":
    defaults = TRAIN_NR_DEFAULTS
  elif mode == "train_sup":
    defaults = TRAIN_SUP_DEFAULTS
  elif mode == "train_unsup":
    defaults = TRAIN_UNSUP_DEFAULTS
  elif mode == "rsd_per_scene":
    defaults = RSD_PER_SCENE_DEFAULTS
  elif mode == "train_sup_per_scene":
    defaults = TRAIN_SUP_PER_SCENE_DEFAULTS
  elif mode == "train_unsup_per_scene":
    defaults = TRAIN_UNSUP_PER_SCENE_DEFAULTS
  elif mode == "test_nr":
    defaults = TEST_NR_DEFAULTS
  elif mode == "test_sup":
    defaults = TEST_SUP_DEFAULTS
  elif mode == "test_unsup":
    defaults = TEST_UNSUP_DEFAULTS
  else:
    raise ValueError("invalid config mode: {:s}".format(mode))
  _merge(defaults, config)
  return config