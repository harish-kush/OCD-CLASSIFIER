import h5py

with h5py.File("lstm_model.h5", "r+") as f:
    config = f.attrs["model_config"]

    if isinstance(config, bytes):
        config = config.decode()

    config = config.replace(
        ', "quantization_config": null',
        ''
    )

    f.attrs.modify("model_config", config)

print("Fixed!")