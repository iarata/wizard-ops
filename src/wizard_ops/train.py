from torch.utils.data import DataLoader
import torch 
from lightning import Trainer
# def train():
    # dataset = MyDataset("data/raw")
    # model = Model()
    # add rest of your training code here

# if __name__ == "__main__":
#     train()


from wizard_ops.data import NutritionDataset, Nutrition, get_default_transforms
from wizard_ops.model import NutritionPredictor

def main() -> None:
    # Use num_workers=0 to avoid macOS spawn guard issues for quick inspection

    transform = get_default_transforms()
#    dataset = NutritionDataset(data_path="src/wizard_ops/data.nosync",frame_idx=2, num_workers=0)
    dataset = NutritionDataset(data_path="src/wizard_ops/data.nosync", frame_idx=1, num_workers=0,transform=transform)
    dataset.setup(stage="fit")

    # train_loader = dataset.train_dataloader()

    # Get first batch and print shapes
    # first_batch = next(iter(train_loader))
    # img = first_batch['image']
    # nutrition = torch.stack([first_batch['calories'], first_batch['mass'], first_batch['fat'], first_batch['carbs'], first_batch['protein'] ],axis=1)
    # print(nutrition.shape)
    # print(nutrition[0])

    model = NutritionPredictor()
    trainer = Trainer(accelerator='mps', max_epochs=5)
    trainer.fit(model, datamodule=dataset)




if __name__ == "__main__":
    main()

