import yaml
from model import Llama3Model
from dataset.dataset import CustomDataset


if __name__ == "__main__":
    with open("training-configs.yaml", encoding="utf8") as conf:
        args = yaml.load(conf, yaml.SafeLoader)

    model = Llama3Model(args["model"], args["custom_checkpoint_path"], **args['training'])
    dataset = CustomDataset("dataset/dset.json")
    model.finetune(dataset, epochs=args['epochs'], **args['optimization'])
    model.save_checkpoint("./checkpoints")
