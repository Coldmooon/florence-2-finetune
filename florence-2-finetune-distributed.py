import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AdamW, get_scheduler
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset


# Initialize the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Create the model
def create_model(rank, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(rank)
    model = DDP(model, device_ids=[rank])
    return model

# Define the dataset class
class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + example['question']
        first_answer = example['answers'][0]
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, first_answer, image

# Create data loaders
def create_data_loaders(rank, world_size, processor, data, batch_size, num_workers):
    # Collate function for data loader
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
        return inputs, answers
    
    train_dataset = DocVQADataset(data['train'])
    val_dataset = DocVQADataset(data['validation'])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader

# Training loop
def train_model(rank, world_size, model_path, processor, data, epochs=10, lr=1e-6, batch_size=1, num_workers=0):
    setup(rank, world_size)
    
    model = create_model(rank, model_path)
    train_loader, val_loader = create_data_loaders(rank, world_size, processor, data, batch_size, num_workers)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            inputs, answers = batch

            input_ids = inputs["input_ids"].to(rank)
            pixel_values = inputs["pixel_values"].to(rank)
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(rank)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Rank {rank}, Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}", leave=False):
                inputs, answers = batch

                input_ids = inputs["input_ids"].to(rank)
                pixel_values = inputs["pixel_values"].to(rank)
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(rank)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Rank {rank}, Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        if rank == 0:
            output_dir = f"./model_checkpoints/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    dist.destroy_process_group()



# Main function to launch training
def main():
    model_path = "microsoft/Florence-2-large-ft"
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load the dataset
    data_dir = "HuggingFaceM4/DocumentVQA"
    data = load_dataset(data_dir)


    world_size = torch.cuda.device_count()
    mp.spawn(train_model,
             args=(world_size, model_path, processor, data),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()

