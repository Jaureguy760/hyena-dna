# import hydra
# from omegaconf import OmegaConf


# def evaluate(config: OmegaConf):
#     raise NotImplementedError("Evaluation not implemented yet")


# @hydra.main(config_path="configs", config_name="config.yaml")
# def main(config: OmegaConf):
#     # Process config:
#     # - register evaluation resolver
#     # - filter out keys used only for interpolation
#     # - optional hooks, including disabling python warnings or debug friendly configuration
#     config = utils.train.process_config(config)

#     # Pretty print config using Rich library
#     utils.train.print_config(config, resolve=True)

#     evaluate(config)


# if __name__ == "__main__":
#     main()
