import os
import json
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from bert.tokenization_bert import BertTokenizer
import torch
import random
import numpy as np

class SalObjDataset(data.Dataset):
    """Dataset for multi-modal salient object detection task."""
    
    def __init__(self, image_root, gt_root, text_file, trainsize, cache_text=False, augment=True):
        """
        Initialize the dataset.
        
        Args:
            image_root (str): Root directory for images
            gt_root (str): Root directory for ground truth masks
            text_file (str): Path to JSON file containing text descriptions
            trainsize (int): Size to which images will be resized
            cache_text (bool): Whether to cache tokenized text descriptions
            augment (bool): Whether to apply data augmentation
        """
        self.trainsize = trainsize
        self.cache_text = cache_text
        self.augment = augment
        
        # Get image and ground truth paths
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Sort to ensure consistent ordering
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # Load text descriptions
#        with open(text_file, 'r', encoding='utf-8') as f:
#            text_data = json.load(f)
        
        # Create filename to description mapping
#        self.text_descriptions = {item['filename']: item['description'] for item in text_data}
        with open(text_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        print(f"Total items in text_data: {len(text_data)}")

        self.text_descriptions = {}
        invalid_items = []
    
        for index, item in enumerate(text_data):
            missing_keys = []
            if 'filename' not in item:
                missing_keys.append('filename')
            if 'description' not in item:
                missing_keys.append('description')

            if missing_keys:
                invalid_items.append({
                    'index': index,
                    'item': item,
                    'missing_keys': missing_keys
                })
                continue

            self.text_descriptions[item['filename']] = item['description']

        if invalid_items:
            print(f"Warning: {len(invalid_items)} invalid items found:")
            for invalid_item in invalid_items:
                print(f"Item at index {invalid_item['index']}:")
                print(f"  Missing keys: {invalid_item['missing_keys']}")
                print(f"  Item details: {invalid_item['item']}")
    
        print(f"Created text descriptions for {len(self.text_descriptions)} files")
        print(f"Number of invalid items: {len(invalid_items)}")
        
        # Initialize transforms
        self.setup_transforms()
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            '/data/dlm/project/multmodel/model/huggingface/bert-base-uncased',
            local_files_only=True
        )
        
        # Filter invalid files and validate data
        self.filter_files()
        validation_results = self.validate_data()
        self.report_validation_results(validation_results)
        
        # Cache tokenized text if requested
        self.cached_text = {}
        if self.cache_text:
            self.cache_text_descriptions()
            
        self.size = len(self.images)

    def setup_transforms(self):
        """Setup image transforms and augmentations."""
        # Basic transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        
        # Augmentation transforms
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])

    def filter_files(self):
        """Filter out invalid files and ensure image-GT pairs match."""
        assert len(self.images) == len(self.gts), \
            f'Number of images ({len(self.images)}) and GT ({len(self.gts)}) do not match!'
            
        valid_images = []
        valid_gts = []
        
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                valid_images.append(img_path)
                valid_gts.append(gt_path)
            else:
                print(f"Size mismatch: {img_path} - {gt_path}")
                
        self.images = valid_images
        self.gts = valid_gts

    def validate_data(self):
        """Validate data integrity."""
        missing_text = []
        missing_images = []
        missing_gt = []
        
        for image_path in self.images:
            image_name = os.path.basename(image_path)
            gt_name = image_name.replace('.jpg', '.png')
            gt_path = os.path.join(os.path.dirname(self.gts[0]), gt_name)
            
            if image_name not in self.text_descriptions:
                missing_text.append(image_name)
            if not os.path.exists(image_path):
                missing_images.append(image_name)
            if not os.path.exists(gt_path):
                missing_gt.append(image_name)
        
        return {
            'missing_text': missing_text,
            'missing_images': missing_images,
            'missing_gt': missing_gt
        }

    def report_validation_results(self, results):
        """Report data validation results."""
        for key, value in results.items():
            if value:
                print(f"Warning: {key}: {len(value)} items")
                for item in value:
                    print(f"Warning: Missing data for {item}")
        print(f"Found {len(self.images)} valid samples out of {len(self.images) + len(results['missing_text'])} images")

    def cache_text_descriptions(self):
        """Pre-compute and cache tokenized text descriptions."""
        print("Pre-computing text tokenization...")
        for image_name, text in self.text_descriptions.items():
            try:
                self.cached_text[image_name] = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )["input_ids"].squeeze(0)
            except Exception as e:
                print(f"Error caching text for {image_name}: {e}")
                self.cached_text[image_name] = torch.zeros(128, dtype=torch.long)

    def augment_data(self, image, gt):
        """Apply augmentation to image and ground truth."""
        if not self.augment:
            return image, gt
            
        # Apply same random flip to both image and GT
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            gt = transforms.functional.hflip(gt)
            
        # Apply color augmentation only to image
        if self.aug_transform:
            image = self.aug_transform(image)
            
        return image, gt

    def __getitem__(self, index):
        """Get a single item from the dataset."""
        image_path = self.images[index]
        gt_path = self.gts[index]
        image_name = os.path.basename(image_path)

        # Load image and GT
        try:
            image = self.rgb_loader(image_path)
            gt = self.binary_loader(gt_path)
        except Exception as e:
            print(f"Error loading images for {image_name}: {e}")
            # Return a black image and mask as fallback
            image = Image.new('RGB', (self.trainsize, self.trainsize))
            gt = Image.new('L', (self.trainsize, self.trainsize))

        # Apply augmentation
        image, gt = self.augment_data(image, gt)

        # Get text description
        if self.cache_text and image_name in self.cached_text:
            encoded_text = self.cached_text[image_name]
        else:
            text = self.text_descriptions.get(image_name, "")
            if not text:
                print(f"Warning: No text description found for {image_name}")
                text = "No description available"
                
            try:
                encoded_text = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )["input_ids"].squeeze(0)
            except Exception as e:
                print(f"Error tokenizing text for {image_name}: {e}")
                encoded_text = torch.zeros(128, dtype=torch.long)

        # Apply transforms
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt, encoded_text

    def rgb_loader(self, path):
        """Load RGB images."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        """Load binary masks."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, text_file, batchsize, trainsize, 
               shuffle=False, num_workers=12, pin_memory=False, augment=False, cache_text=False):
    """Create data loader with specified parameters."""
    
    dataset = SalObjDataset(
        image_root=image_root,
        gt_root=gt_root,
        text_file=text_file,
        trainsize=trainsize,
        augment=augment,
        cache_text=cache_text
    )
    
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return data_loader

class test_dataset:
    """Dataset class for testing/inference."""
    
    def __init__(self, image_root, gt_root, text_file, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        # Load text descriptions
#        with open(text_file, 'r', encoding='utf-8') as f:
#            text_data = json.load(f)
#        self.text_descriptions = {item['filename']: item['description'] for item in text_data}
        with open(text_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        print(f"Total items in text_data: {len(text_data)}")

        self.text_descriptions = {}
        invalid_items = []
    
        for index, item in enumerate(text_data):
            missing_keys = []
            if 'filename' not in item:
                missing_keys.append('filename')
            if 'description' not in item:
                missing_keys.append('description')

            if missing_keys:
                invalid_items.append({
                    'index': index,
                    'item': item,
                    'missing_keys': missing_keys
                })
                continue

            self.text_descriptions[item['filename']] = item['description']

        if invalid_items:
            print(f"Warning: {len(invalid_items)} invalid items found:")
            for invalid_item in invalid_items:
                print(f"Item at index {invalid_item['index']}:")
                print(f"  Missing keys: {invalid_item['missing_keys']}")
                print(f"  Item details: {invalid_item['item']}")
    
        print(f"Created text descriptions for {len(self.text_descriptions)} files")
        print(f"Number of invalid items: {len(invalid_items)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        
        self.tokenizer = BertTokenizer.from_pretrained(
            '/data/dlm/project/multmodel/model/huggingface/bert-base-uncased',
            local_files_only=True
        )
        
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        """Load next test sample."""
        image_path = self.images[self.index]
        gt_path = self.gts[self.index]
        image_name = os.path.basename(image_path)
        
        # Load image and GT
        image = self.rgb_loader(image_path)
        gt = self.binary_loader(gt_path)
        
        # Get text description
        text = self.text_descriptions.get(image_name, "")
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )["input_ids"]
        
        # Apply transforms
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        
        # Get image name without extension
        name = os.path.splitext(image_name)[0] + '.png'
        
        self.index += 1
        return image, gt, encoded_text, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.images)