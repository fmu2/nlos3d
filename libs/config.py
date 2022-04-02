import yaml


RSD_DEFAULTS = {
  "dataset": {
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

  "batch_size": 1,
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


TRAIN_NR_DEFAULTS = {
  "dataset": {
    "ds": 1,
    "scale": 1,
    "background": 0,
    "target_noise": 0,
  },

  "model": {
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
    "n_itrs": 50000,

    "optim_type": "adam",
    "lr": 1e-4,
    "weight_decay": 0,
    "beta1": 0.9,
    "beta2": 0.999,

    "clip_grad_norm": 1,

    "sched_type": "step",
    "milestones": [40000],
    "gamma": 0.1,
  },

  "train": {
    "n_views": 4,
    "include_orthogonal": True,
    "in_scale": 1,
  },

  "eval": {
    "n_views": 1,
    "include_orthogonal": True,
    "in_scale": 1,
  },
}


TRAIN_SUP_PER_SCENE_DEFAULTS = {
  "ckpt_name": "num2",
  "save_freq": 1000,
  "print_freq": 100,
  "diagnose": False,

  "dataset": {
    "path": "./data/per_scene/num2/trial_1/1_frame.mat",
    "scale": 1,
    "clip": 200,
  },

  "add_noise": True,
  "noise": {
    "path": "./data/per_scene/noise/noise.mat",
    "scale": 1,
    "clip": 200,
  },

  "camera": {   # NOTE: z-axis points into the camera
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
      "res": 256,
      "block_size": 4,
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
        "n_freqs": 10,
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
        "hid_dim": 256,
        "skips": [5],
        "n_sigma_layers": 8,
        "n_color_layers": 1,
        "actv": "relu",
      },
      
      "nsvf": {
        "hid_dim": 256,
        "n_sigma_layers": 4,
        "n_color_layers": 4,
        "actv": "relu",
      },

      "siren": {
        "hid_dim": 256,
        "n_sigma_layers": 8,
        "n_color_layers": 1,
        "in_w0": 30,
        "hid_w0": 30,
      },
    },
    
    "common": {
      "bb_ctr": [0, 0, 0.3],
      "bb_size": [2, 2, 1],
      "inf": 10,
      "p_polar": False,
      "d_polar": True,
      "z_norm": False,
      "sigma_transform": "relu",
      "color_transform": "relu",
    },

    "steady_state": {
      "bin_len": 0.03,
      "n_bins": 48,
    },
  },

  "loss": {
    "mb": 1,
    "md": 0,
    "bp": 1e-4,
    "tva": 1e-5, 
  },

  "optim": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
      "momentum": 0.9,
      "weight_decay": 0,
      "beta1": 0.9,
      "beta2": 0.999,
    },
    
    "scheduler": {
      "type": "exp",
      "milestone": [0.8],
      "gamma": 0.1,
      "decay_steps": 10000,
    },
  },

  "main": {
    "n_iters": 5000,

    "n_views": 4,
    "include_orthogonal": False,

    "train": {
      "n_steps": 2,
      "s_scale": 1,
      "sigma_noise": 1,
      "color_noise": 0,
    },

    "val": {
      "n_steps": 2,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
  },
}


TRAIN_UNSUP_PER_SCENE_DEFAULTS = {
  "ckpt_name": "num2",
  "save_freq": 1000,
  "print_freq": 100,
  "diagnose": False,

  "dataset": {
    "path": "./data/per_scene/num2/trial_1/1_frame.mat",
    "scale": 1,
    "clip": 200,
  },

  "add_noise": True,
  "noise": {
    "path": "./data/per_scene/noise/noise.mat",
    "scale": 1,
    "clip": 200,
  },

  "camera": {   # NOTE: z-axis points into the camera
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
      "res": 256,
    },
  },

  "wall": {
    "size": 2,
    "res": 64,
    "block_size": 32,
    "uv_size": 64,
    "foreshortening": False,
    "depth": 0.8,
  },

  "model": {
    "embedder": {
      "embed_p": True,
      "embed_d": True,
      "embed_z": False,

      "p": {
        "in_dim": 3,
        "include_input": True,
        "n_freqs": 10,
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
        "hid_dim": 256,
        "skips": [5],
        "n_sigma_layers": 8,
        "n_color_layers": 1,
        "actv": "relu",
      },
      
      "nsvf": {
        "hid_dim": 256,
        "n_sigma_layers": 4,
        "n_color_layers": 4,
        "actv": "relu",
      },

      "siren": {
        "hid_dim": 256,
        "n_sigma_layers": 8,
        "n_color_layers": 1,
        "in_w0": 30,
        "hid_w0": 30,
      },
    },
    
    "common": {
      "bb_ctr": [0, 0, 0.3],
      "bb_size": [2, 2, 1],
      "inf": 10,
      "p_polar": False,
      "d_polar": True,
      "z_norm": False,
      "sigma_transform": "relu",
      "color_transform": "relu",
    },

    "transient": {
      "light": [0, 0, -1],
      "bin_range": [50, 150],
      "bin_len": 0.03,
    },

    "steady_state": {
      "bin_len": 0.03,
      "n_bins": 48,
    },
  },

  "loss": {
    "pm": 1,
    "bp": 1e-4,
    "tva": 1e-5, 
  },

  "optim": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
      "momentum": 0.9,
      "weight_decay": 0,
      "beta1": 0.9,
      "beta2": 0.999,
    },
    
    "scheduler": {
      "type": "exp",
      "milestone": [0.8],
      "gamma": 0.1,
      "decay_steps": 10000,
    },
  },

  "main": {
    "n_iters": 5000,

    "n_views": 4,
    "include_orthogonal": True,

    "train": {
      "n_steps": 2,
      "t_scale": 1,
      "s_scale": 1,
      "sigma_noise": 1,
      "color_noise": 0,
    },

    "val": {
      "n_steps": 2,
      "t_scale": 1,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
  },
}

TRAIN_DEFAULTS = {
  "ckpt_name": "alphanumeric",
  "save_freq": 10,    # save every X epochs
  "train_freq": 100,  # print every Y iterations
  "val_freq": 10,
  "diagnose": False,

  "dataset": {
    "root": "./data/alphanumeric",
    "scale": 1,
    "clip": 200,
    "background": 0,
    "target_size": 256,
  },

  "camera": {   # NOTE: z-axis points into the camera
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
      "res": 256,
      "block_size": 4,
    },
  },

  "wall": {
    "size": 2,
    "res": 64,
    "block_size": 32,
    "uv_size": 64,
    "foreshortening": False,
    "depth": 0.8,
  },

  "model": {
    "encoder": {
      "type": "densenet",

      "densenet": {
        "in_plane": 3,
        "plane": 64,
        "n_blocks": 3,
        "n_layers": 4,
        "layer_type": "v3",
        "growth_rate": 16,
        "bn": [True, True, False],
        "ds": [True, True, False],
        "actv": "leaky_relu",
        "norm": "instance",
        "affine": True,
      }
    },
    
    "renderer": {
      "embedder": {
        "embed_p": True,
        "embed_d": True,
        "embed_z": False,

        "p": {
          "in_dim": 3,
          "include_input": True,
          "n_freqs": 10,
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
          "in_dim": 6,
          "include_input": True,
          "n_freqs": 6,
          "log_sampling": True,
          "freq_scale": 1,
        },
      },
      
      "field": {
        "type": "nerf",

        "nerf": {
          "hid_dim": 32,
          "skips": [],
          "n_sigma_layers": 1,
          "n_color_layers": 1,
          "actv": "relu",
          "film_hid_dim": 16,
          "film_n_layers": 2,
          "film_actv": "leaky_relu",
        },
      },

      "common": {
        "bb_ctr": [0, 0, 0.3],
        "bb_size": [2, 2, 1],
        "inf": 10,
        "p_polar": False,
        "d_polar": True,
        "z_norm": False,
        "sigma_transform": "relu",
        "color_transform": "relu",
      },

      "transient": {
        "light": [0, 0, -1],
        "bin_range": [50, 150],
        "bin_len": 0.03,
      },

      "steady_state": {
        "bin_len": 0.03,
        "n_bins": 48,
      },
    },
  },

  "loss": {
    "pm": 1e-3,
    "mb": 1,
    "md": 0,
    "bp": 1e-4,
    "tva": 1e-5, 
  },

  "optim": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
      "momentum": 0.9,
      "weight_decay": 0,
      "beta1": 0.9,
      "beta2": 0.999,
    },
    
    "scheduler": {
      "type": "exp",
      "milestone": [0.8],
      "gamma": 0.1,
      "decay_steps": 25000,
    },
  },

  "main": {
    "n_epochs": 25,
    "batch_size": 2,
    "n_workers": 6,

    "n_views": 4,
    "include_orthogonal": True,

    "train": {
      "in_scale": None,
      "n_steps": 2,
      "t_scale": 25465,
      "s_scale": 1,
      "sigma_noise": 1,
      "color_noise": 0,
    },

    "val": {
      "in_scale": None,
      "n_steps": 2,
      "t_scale": 25465,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
  },
}


TRAIN_UNSUP_DEFAULTS = {
  "ckpt_name": "alphanumeric",
  "save_freq": 10,    # save every X epochs
  "train_freq": 100,  # print every Y iterations
  "val_freq": 10,
  "diagnose": False,

  "dataset": {
    "root": "./data/alphanumeric",
    "scale": 1,
    "clip": 200,
    "background": [0.1, 0.5],
  },

  "camera": {   # NOTE: z-axis points into the camera
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
      "res": 256,
    },
  },

  "wall": {
    "size": 2,
    "res": 64,
    "block_size": 32,
    "uv_size": 64,
    "foreshortening": False,
    "depth": 0.8,
  },

  "model": {
    "encoder": {
      "type": "densenet",

      "densenet": {
        "in_plane": 3,
        "plane": 64,
        "n_blocks": 3,
        "n_layers": 4,
        "layer_type": "v3",
        "growth_rate": 16,
        "bn": [True, True, False],
        "ds": [True, True, False],
        "actv": "leaky_relu",
        "norm": "instance",
        "affine": True,
      }
    },
    
    "renderer": {
      "embedder": {
        "embed_p": True,
        "embed_d": True,
        "embed_z": False,

        "p": {
          "in_dim": 3,
          "include_input": True,
          "n_freqs": 10,
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
          "in_dim": 6,
          "include_input": True,
          "n_freqs": 6,
          "log_sampling": True,
          "freq_scale": 1,
        },
      },
      
      "field": {
        "type": "nerf",

        "nerf": {
          "hid_dim": 32,
          "skips": [],
          "n_sigma_layers": 1,
          "n_color_layers": 1,
          "actv": "relu",
          "film_hid_dim": 16,
          "film_n_layers": 2,
          "film_actv": "leaky_relu",
        },
      },

      "common": {
        "bb_ctr": [0, 0, 0.3],
        "bb_size": [2, 2, 1],
        "inf": 10,
        "p_polar": False,
        "d_polar": True,
        "z_norm": False,
        "sigma_transform": "relu",
        "color_transform": "relu",
      },

      "transient": {
        "light": [0, 0, -1],
        "bin_range": [50, 150],
        "bin_len": 0.03,
      },

      "steady_state": {
        "bin_len": 0.03,
        "n_bins": 48,
      },
    },
  },

  "loss": {
    "pm": 1,
    "bp": 1e-4,
    "tva": 1e-5, 
  },

  "optim": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
      "momentum": 0.9,
      "weight_decay": 0,
      "beta1": 0.9,
      "beta2": 0.999,
    },
    
    "scheduler": {
      "type": "exp",
      "milestone": [0.8],
      "gamma": 0.1,
      "decay_steps": 25000,
    },
  },

  "main": {
    "n_epochs": 25,
    "batch_size": 2,
    "n_workers": 6,

    "n_views": 4,
    "include_orthogonal": True,

    "train": {
      "in_scale": None,
      "n_steps": 2,
      "t_scale": 25465,
      "s_scale": 1,
      "sigma_noise": 1,
      "color_noise": 0,
    },

    "val": {
      "in_scale": None,
      "n_steps": 2,
      "t_scale": 25465,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
  },
}


TRAIN_SUP_DEFAULTS = {
  "ckpt_name": "alphanumeric",
  "save_freq": 10,    # save every X epochs
  "train_freq": 100,  # print every Y iterations
  "val_freq": 10,
  "diagnose": False,

  "dataset": {
    "root": "./data/alphanumeric",
    "scale": 1,
    "clip": 200,
    "background": [0.1, 0.5],
    "target_size": 256,
  },

  "camera": {   # NOTE: z-axis points into the camera
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
      "res": 256,
      "block_size": 4,
    },
  },

  "model": {
    "encoder": {
      "type": "densenet",

      "densenet": {
        "in_plane": 3,
        "plane": 64,
        "n_blocks": 3,
        "n_layers": 4,
        "layer_type": "v3",
        "growth_rate": 16,
        "bn": [True, True, False],
        "ds": [True, True, False],
        "actv": "leaky_relu",
        "norm": "instance",
        "affine": True,
      }
    },
    
    "renderer": {
      "embedder": {
        "embed_p": True,
        "embed_d": True,
        "embed_z": False,

        "p": {
          "in_dim": 3,
          "include_input": True,
          "n_freqs": 10,
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
          "in_dim": 16,
          "include_input": True,
          "n_freqs": 6,
          "log_sampling": True,
          "freq_scale": 1,
        },
      },
      
      "field": {
        "type": "nerf",

        "nerf": {
          "hid_dim": 32,
          "skips": [],
          "n_sigma_layers": 1,
          "n_color_layers": 1,
          "actv": "relu",
          "film_hid_dim": 16,
          "film_n_layers": 2,
          "film_actv": "leaky_relu",
        },
      },

      "common": {
        "bb_ctr": [0, 0, 0.3],
        "bb_size": [2, 2, 1],
        "inf": 10,
        "p_polar": False,
        "d_polar": True,
        "z_norm": False,
        "sigma_transform": "relu",
        "color_transform": "relu",
      },

      "steady_state": {
        "bin_len": 0.03,
        "n_bins": 48,
      },
    },
  },

  "loss": {
    "mb": 1,
    "md": 0,
    "bp": 1e-4,
    "tva": 1e-5, 
  },

  "optim": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-4,
      "momentum": 0.9,
      "weight_decay": 0,
      "beta1": 0.9,
      "beta2": 0.999,
    },
    
    "scheduler": {
      "type": "exp",
      "milestone": [0.8],
      "gamma": 0.1,
      "decay_steps": 25000,
    },
  },

  "main": {
    "n_epochs": 25,
    "batch_size": 2,
    "n_workers": 6,

    "n_views": 4,
    "include_orthogonal": True,

    "train": {
      "in_scale": 1,
      "n_steps": 3,
      "s_scale": 1,
      "sigma_noise": 1,
      "color_noise": 0,
    },

    "val": {
      "in_scale": 1,
      "n_steps": 3,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
  },
}


TEST_DEFAULTS = {
  "ckpt": "last.pth",

  "dataset": {
    "root": "./data/alphanumeric-simple",
    "clip": 200,
    "scale": 1,
    "background": 0,
    "target_size": 256,
    "target_noise": 0,
  },

  "camera": {
    "type": "orthographic",
      
    "orthographic": {
      "size": 2,
      "res": 256,
    },
  },

  "main": {
    "batch_size": 4,
    "n_workers": 6,

    "all_views": False,

    "render": {
      "in_scale": 1,
      "n_steps": 2,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
  },
}

TEST_PER_SCENE_DEFAULTS = {
  "ckpt": "last.pth",

  "measurement": {
    "clip": 200,
    "scale": 1,
    "background": 0,
  },

  "camera": {
    "type": "orthographic",
    
    "orthographic": {
      "size": 2,
      "res": 256,
    },
  },

  "main": {
    "all_views": False,

    "render": {
      "in_scale": 1,
      "n_steps": 2,
      "s_scale": 1,
      "sigma_noise": 0,
      "color_noise": 0,
    },
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
  if mode == "train_nr":
    config = TRAIN_NR_DEFAULTS
  elif mode == "train":
    config = TRAIN_DEFAULTS
  elif mode == "train_unsup":
    config = TRAIN_UNSUP_DEFAULTS
  elif mode == "train_sup":
    config = TRAIN_SUP_DEFAULTS
  elif mode == "train_sup_per_scene":
    config = TRAIN_SUP_PER_SCENE_DEFAULTS
  elif mode == "train_unsup_per_scene":
    config = TRAIN_UNSUP_PER_SCENE_DEFAULTS
  elif mode == "test":
    config = TEST_DEFAULTS
  elif mode == "test_per_scene":
    config = TEST_PER_SCENE_DEFAULTS
  elif mode == "rsd":
    config = RSD_DEFAULTS
  elif mode == "rsd_per_scene":
    config = RSD_PER_SCENE_DEFAULTS
  else:
    raise ValueError("invalid default config mode: {:s}".format(mode))
  return config


def load_config(config_file, mode="train"):
  with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  if mode == "train_nr":
    defaults = TRAIN_NR_DEFAULTS
  elif mode == "train":
    defaults = TRAIN_DEFAULTS
  elif mode == "train_unsup":
    defaults = TRAIN_UNSUP_DEFAULTS
  elif mode == "train_sup":
    defaults = TRAIN_SUP_DEFAULTS
  elif mode == "train_sup_per_scene":
    defaults = TRAIN_SUP_PER_SCENE_DEFAULTS
  elif mode == "train_unsup_per_scene":
    defaults = TRAIN_UNSUP_PER_SCENE_DEFAULTS
  elif mode == "test":
    defaults = TEST_DEFAULTS
  elif mode == "test_per_scene":
    defaults = TEST_PER_SCENE_DEFAULTS
  elif mode == "rsd":
    defaults = RSD_DEFAULTS
  elif mode == "rsd_per_scene":
    defaults = RSD_PER_SCENE_DEFAULTS
  else:
    raise ValueError("invalid config mode: {:s}".format(mode))
  _merge(defaults, config)
  return config