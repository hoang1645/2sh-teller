import yaml
from model import Llama3Model
from dataset.dataset import CustomDataset

# def parse_args():
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_path', type=str, required=False, default=None)
# parser.add_argument("--model_size", choices=["8B", "8B-Instruct", "70B", "70B-Instruct"], default="8B-Instruct")
# parser.add_argument('--lr', type=float, required=False, default=2e-4, help='learning rate')
# parser.add_argument('--batch_size', type=int, required=False, default=4, help='batch size')
# parser.add_argument('--beta1', type=float, required=False, default=.9, help='adam optimization algorithm\'s β1')
# parser.add_argument('--beta2', type=float, required=False, default=.99, help='adam optimization algorithm\'s β2')
# parser.add_argument('--autocast', action=argparse.BooleanOptionalAction, required=False, help='use automatic type casting')
# parser.add_argument('--from_ckpt', type=str, required=False, default=None, help='load model from checkpoint at specified path')
# parser.add_argument('--epoch', type=int, required=False, default=100, help='number of training epochs')
# # parser.add_argument('--update_discriminator_every_n_steps', type=int, required=False, default=1)
# return parser.parse_args()

if __name__ == "__main__":
    with open("training-configs.yaml", encoding="utf8") as conf:
        args = yaml.load(conf, yaml.SafeLoader)

    model = Llama3Model(args["model_size"], args["from_ckpt"], **args['training'])
    dataset = CustomDataset("dataset/dset.json")
    model.finetune(dataset, epochs=args['epochs'], **args['optimization'])
    model.save_checkpoint("./checkpoints")
