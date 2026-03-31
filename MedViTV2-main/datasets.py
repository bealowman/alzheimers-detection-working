import os
import json
from torch.utils.data import DataLoader, random_split, Subset
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import medmnist
from medmnist import INFO, Evaluator


import requests
from zipfile import ZipFile
import pandas as pd
import shutil

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)

from zipfile import ZipFile
import pandas as pd
import shutil

from torchvision import datasets
from sklearn.model_selection import StratifiedKFold

try:
    from google.colab import userdata
    import kaggle
except ImportError:
    pass

root_dir='data'
if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
class PADatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'zr7vgbcyr2-1.zip')
        self.dataset_extracted_dir = self.root_dir
        self.source_images_dirs = [os.path.join(self.root_dir, 'images', f'imgs_part_{i}') for i in range(1, 4)]
        self.organized_images_dir = os.path.join(self.root_dir, 'PAD-Dataset')
        self.metadata_file_path = os.path.join(self.root_dir, 'metadata.csv')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(os.path.join(self.root_dir, 'images')):
            print("Extracting main dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Main extraction complete.")
        
    def extract_inner_datasets(self):
        for i, source_images_dir in enumerate(self.source_images_dirs, start=1):
            inner_zip_path = os.path.join(self.root_dir, f'images/imgs_part_{i}.zip')
            if not os.path.exists(source_images_dir):
                print(f"Extracting {inner_zip_path}...")
                with ZipFile(inner_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(source_images_dir))
                print(f"Extraction of {inner_zip_path} complete.")
        
    def organize_images(self):
        if os.path.exists(self.organized_images_dir):
            print("Images are already organized.")
            return
        
        if not os.path.exists(self.metadata_file_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file_path}")
        
        metadata = pd.read_csv(self.metadata_file_path)
        
        os.makedirs(self.organized_images_dir, exist_ok=True)
        
        diagnostic_labels = metadata['diagnostic'].unique()
        
        for label in diagnostic_labels:
            os.makedirs(os.path.join(self.organized_images_dir, label), exist_ok=True)
        
        for _, row in metadata.iterrows():
            img_id = row['img_id']
            diagnostic = row['diagnostic']
            
            for source_dir in self.source_images_dirs:
                source_path = os.path.join(source_dir, img_id)
                if os.path.exists(source_path):
                    destination_path = os.path.join(self.organized_images_dir, diagnostic, img_id)
                    shutil.move(source_path, destination_path)
                    break
        
        print("Images moved successfully.")
        
    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_datasets()
            self.organize_images()
            return self.organized_images_dir



class FetalDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'FETAL_PLANES_ZENODO.zip')
        self.dataset_extracted_dir = self.root_dir
        self.organized_images_dir = os.path.join(self.root_dir, 'Fetal-Dataset')
        self.excel_file_path = os.path.join(self.root_dir, 'FETAL_PLANES_DB_data.xlsx')
        self.source_images_dir = os.path.join(self.root_dir, 'Images')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(self.excel_file_path) or not os.path.exists(self.source_images_dir):
            print("Extracting dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction complete.")
        
    def organize_images(self):
        if os.path.exists(self.organized_images_dir):
            print("Images are already organized.")
            return
        
        if not os.path.exists(self.excel_file_path):
            raise FileNotFoundError(f"Excel file not found at {self.excel_file_path}")
        
        df = pd.read_excel(self.excel_file_path)
        
        os.makedirs(self.organized_images_dir, exist_ok=True)
        
        plane_labels = df['Plane'].unique()
        
        for label in plane_labels:
            os.makedirs(os.path.join(self.organized_images_dir, str(label)), exist_ok=True)
        
        for _, row in df.iterrows():
            img_id = row['Image_name']
            plane = row['Plane']
            source_path = os.path.join(self.source_images_dir, f'{img_id}.png')
            destination_path = os.path.join(self.organized_images_dir, str(plane), f'{img_id}.png')
            
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
        
        print("Images moved successfully.")
        
    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.organize_images()
            return self.organized_images_dir


class ISICDatasetManager:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.train_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip'
        self.test_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip'
        self.train_gt_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip'
        self.test_gt_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip'
        self.train_path = os.path.join(self.base_dir, 'ISIC2018_Train')
        self.test_path = os.path.join(self.base_dir, 'ISIC2018_Test')

        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)

    def download_and_extract(self, url, extract_to):
        local_filename = os.path.join(self.base_dir, url.split('/')[-1])
        if not os.path.exists(local_filename):
            print(f"Downloading {url}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
        print(f"Extracting {local_filename}...")
        with ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")

    def organize_by_labels(self, metadata_path, image_dir, output_base_dir):
        metadata = pd.read_csv(metadata_path)
        labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        for label in labels:
            os.makedirs(os.path.join(output_base_dir, label), exist_ok=True)

        def move_image(row):
            image_name = f"{row['image']}.jpg"
            source_path = os.path.join(image_dir, image_name)
            for label in labels:
                if row[label] == 1.0:
                    target_path = os.path.join(output_base_dir, label, image_name)
                    shutil.move(source_path, target_path)
                    break
        metadata.apply(move_image, axis=1)

    def setup_dataset(self):
        
        # Organize training and test images
        train_categorized = os.path.join(self.train_path, 'Categorized')
        test_categorized = os.path.join(self.test_path, 'Categorized')
        
        if os.path.exists(train_categorized):
            print("Dataset already exists. Returning the root directory.")
            return train_categorized, test_categorized
        else:
            # Download and extract training and test datasets
            self.download_and_extract(self.train_url, self.train_path)
            self.download_and_extract(self.test_url, self.test_path)
            self.download_and_extract(self.train_gt_url, self.train_path)
            self.download_and_extract(self.test_gt_url, self.test_path)

            
            self.organize_by_labels(
                os.path.join(self.train_path, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'),
                os.path.join(self.train_path, 'ISIC2018_Task3_Training_Input'),
                train_categorized
            )
            self.organize_by_labels(
                os.path.join(self.test_path, 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'),
                os.path.join(self.test_path, 'ISIC2018_Task3_Test_Input'),
                test_categorized
            )

            return train_categorized, test_categorized


class CPNDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/dvntn9yhd2-1.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'dvntn9yhd2-1.zip')
        self.dataset_extracted_dir = os.path.join(self.root_dir, 'dvntn9yhd2-1')
        self.organized_images_dir = os.path.join(self.root_dir, 'CPN-Dataset')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(self.dataset_extracted_dir):
            print("Extracting main dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Main extraction complete.")
        
    def extract_inner_dataset(self):
        inner_zip_path = os.path.join(self.dataset_extracted_dir, 'Covid19-Pneumonia-Normal Chest X-Ray Images Dataset.zip')
        if not os.path.exists(self.organized_images_dir):
            os.makedirs(self.organized_images_dir)
            print("Extracting inner dataset...")
            with ZipFile(inner_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.organized_images_dir)
            print("Inner extraction complete.")
        
    def get_dataset(self):
        if os.path.exists(self.organized_images_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.organized_images_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_dataset()
            return self.organized_images_dir


class KvasirDatasetDownloader:
    def __init__(self, root_dir='data', dataset_url='https://datasets.simula.no/downloads/kvasir/kvasir-dataset.zip'):
        self.root_dir = root_dir
        self.dataset_url = dataset_url
        self.dataset_zip_path = os.path.join(self.root_dir, 'kvasir-dataset.zip')
        self.dataset_dir = os.path.join(self.root_dir, 'kvasir-dataset')
        
    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
            
        if not os.path.exists(self.dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            with requests.get(self.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(self.dataset_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        
    def extract_dataset(self):
        if not os.path.exists(self.dataset_dir):
            print("Extracting dataset...")
            with ZipFile(self.dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction complete.")
        
    def get_dataset(self):
        if os.path.exists(self.dataset_dir):
            print("Dataset already exists. Returning the root directory.")
            return self.dataset_dir
        else:
            self.download_dataset()
            self.extract_dataset()
            return self.dataset_dir



        
        
def build_dataset(args):
    train_transform, test_transform = build_transform(args)
    #data_dir = args.dataset_dir
    
    
    if args.dataset == 'Kvasir':
        # Define the sizes for the splits
        train_size = 2408
        val_size = 392
        test_size = 1200
        nb_classes = 8
        downloader = KvasirDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")  
    elif args.dataset == 'CPN':
        # Define the sizes for the splits
        train_size = 3140
        val_size = 521
        test_size = 1567
        nb_classes = 3
        downloader = CPNDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'Fetal':
        # Define the sizes for the splits
        train_size = 7446
        val_size = 1237
        test_size = 3717
        nb_classes = 6
        downloader = FetalDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'PAD':
        # Define the sizes of each split
        train_size = 1384
        val_size = 227
        test_size = 687
        nb_classes = 6
        downloader = PADatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'ISIC2018':
        nb_classes = 7
        manager = ISICDatasetManager()
        train_path, test_path = manager.setup_dataset()
        print(f"Dataset is available at: {train_path}")
        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform) 
        test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform) 
        return train_dataset, test_dataset, nb_classes
    elif args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        n_channels = info['n_channels']
        nb_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        print("Number of channels: ", n_channels)
        print("Number of classes: ", nb_classes)
        train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')
        test_dataset = DataClass(split='test', transform=test_transform, download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')
        return train_dataset, test_dataset, nb_classes
    else:
        raise NotImplementedError()
    
    full_dataset = datasets.ImageFolder(root=data_dir)  # Load without transform
    # Verify the total number of images matches the sum of the splits
    assert train_size + val_size + test_size == len(full_dataset), "The sum of the splits must equal the total number of images"

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply the transformations
    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_dataset.indices)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), val_dataset.indices)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), test_dataset.indices)

    print("Number of the class = %d" % nb_classes)

    return train_dataset, test_dataset, nb_classes


def build_transform(args):
    t_train = []
    # this should always dispatch to transforms_imagenet_train
    t_train.append(transforms.RandomResizedCrop(224))
    t_train.append(transforms.AugMix(alpha= 0.4))
    #t_train.append(transforms.Lambda(lambda image: image.convert('RGB')))
    t_train.append(transforms.RandomHorizontalFlip(p=0.4))
    t_train.append(transforms.ToTensor())
    t_train.append(transforms.Normalize(mean=[.5], std=[.5]))
        

    t_test = []
    t_test.append(transforms.Resize((224, 224)))
    #t_test.append(transforms.Lambda(lambda image: image.convert('RGB')))
    t_test.append(transforms.ToTensor())
    t_test.append(transforms.Normalize(mean=[.5], std=[.5]))
    return transforms.Compose(t_train), transforms.Compose(t_test)


class AlzheimerKFoldManager:
    def __init__(self, root_dir='/content/drive/MyDrive/SP26_dementia_data', n_splits=5, seed=42):
        self.root_dir = root_dir
        self.dataset_id = 'aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented'
        # may need to change based on file structure
        self.data_path = os.path.join(self.root_dir, 'kfoldable/Alzheimer-Dataset/combined_images')
        self.n_splits = n_splits
        self.seed = seed
        
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

    def _setup_kaggle(self):
        #Authenticates Kaggle
        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

    def download_and_prepare(self):
        #check if folder exists first
        # I downloaded the data directly from kaggle 
        # into gDrive so it's already there
        if os.path.exists(self.data_path):
            print(f"Found existing data at {self.data_path}. Skipping download.")
            return self.data_path
        
        #if not, create the path and download
        print("Data not found in Drive. Preparing to download...")
        os.makedirs(self.root_dir, exist_ok=True)
        
        self._setup_kaggle()
        kaggle.api.dataset_download_files(self.dataset_id, path=self.root_dir, unzip=False)
    
        zip_path = os.path.join(self.root_dir, 'alzheimers-multiclass-dataset-equal-and-augmented.zip')
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)
        
        os.remove(zip_path)
        return self.data_path

    def get_folds(self, transform=None):
        full_dataset = datasets.ImageFolder(root=self.data_path, transform=transform)
        
        labels = full_dataset.targets
        indices = list(range(len(full_dataset)))
        
        folds = []
        for train_idx, val_idx in self.skf.split(indices, labels):
            folds.append((Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)))
            
        return folds

    def get_fold_loaders(self, fold_index, transform=None, batch_size=32, num_workers=2):
   
        folds = self.get_folds(transform=transform)
        
        #pick specific fold requested
        train_subset, val_subset = folds[fold_index]
        
        #DataLoaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True 
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    

#MOSTLY BEA'S CODE EXCEPT I WANT TO OVERRIDE TO GOOGLE DRIVE

class DementiaDatasetDownloader:
    def __init__(self, root_dir='data', drive_dir=None):
        self.root_dir = root_dir
        self.drive_dir = drive_dir or os.environ.get('DEMENTIA_DATASET_DIR')
        self.organized_images_dir = os.path.join(self.root_dir, 'Dementia-Dataset')
        self.local_train_dir = os.path.join(self.organized_images_dir, 'train')
        self.local_val_dir = os.path.join(self.organized_images_dir, 'val')

    def _resolve_drive_split_dirs(self):
        if self.drive_dir is None:
            raise ValueError(
                "Dementia dataset path not set. Provide drive_dir=... pointing to a folder containing "
                "'train/' and 'val/' (or set env var DEMENTIA_DATASET_DIR)."
            )

        drive_dir = os.path.expanduser(self.drive_dir)

        base = drive_dir

        drive_train_dir = os.path.join(base, 'train')
        drive_val_dir = os.path.join(base, 'val')

        if not os.path.isdir(drive_train_dir):
            raise FileNotFoundError(f"Expected train directory at: {drive_train_dir}")
        if not os.path.isdir(drive_val_dir):
            raise FileNotFoundError(f"Expected val directory at: {drive_val_dir}")

        return drive_train_dir, drive_val_dir

    def download_dataset(self):
        drive_train_dir, drive_val_dir = self._resolve_drive_split_dirs()
        
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        os.makedirs(self.organized_images_dir, exist_ok=True)

        # Copy per-split so we can safely resume partial copies.
        if not os.path.exists(self.local_train_dir):
            print(f"Copying Dementia train split: {drive_train_dir} -> {self.local_train_dir}")
            shutil.copytree(drive_train_dir, self.local_train_dir)
            print("Train copy complete.")
        else:
            print("Train split already exists locally. Skipping train copy.")

        if not os.path.exists(self.local_val_dir):
            print(f"Copying Dementia val split: {drive_val_dir} -> {self.local_val_dir}")
            shutil.copytree(drive_val_dir, self.local_val_dir)
            print("Val copy complete.")
        else:
            print("Val split already exists locally. Skipping val copy.")

    def extract_dataset(self):
        # Data is pre-organized on Drive, no extraction needed.
        # Included for API consistency with other downloaders.
        pass

    def get_dataset(self):
        drive_train_dir, drive_val_dir = self._resolve_drive_split_dirs()
        
        print(f"Bypassing copy. Training directly from Drive: {drive_train_dir}")
        
        return drive_train_dir, drive_val_dir


        
        
def build_dataset(args):
    train_transform, test_transform = build_transform(args)
    #data_dir = args.dataset_dir
    
    
    if args.dataset == 'Kvasir':
        # Define the sizes for the splits
        train_size = 2408
        val_size = 392
        test_size = 1200
        nb_classes = 8
        downloader = KvasirDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")  
    elif args.dataset == 'CPN':
        # Define the sizes for the splits
        train_size = 3140
        val_size = 521
        test_size = 1567
        nb_classes = 3
        downloader = CPNDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'Fetal':
        # Define the sizes for the splits
        train_size = 7446
        val_size = 1237
        test_size = 3717
        nb_classes = 6
        downloader = FetalDatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'PAD':
        # Define the sizes of each split
        train_size = 1384
        val_size = 227
        test_size = 687
        nb_classes = 6
        downloader = PADatasetDownloader()
        data_dir = downloader.get_dataset()
        print(f"Dataset is available at: {data_dir}")
    elif args.dataset == 'ISIC2018':
        nb_classes = 7
        manager = ISICDatasetManager()
        train_path, test_path = manager.setup_dataset()
        print(f"Dataset is available at: {train_path}")
        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform) 
        test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform) 
        return train_dataset, test_dataset, nb_classes
    elif args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        n_channels = info['n_channels']
        nb_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        print("Number of channels: ", n_channels)
        print("Number of classes: ", nb_classes)
        train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')
        test_dataset = DataClass(split='test', transform=test_transform, download=True, as_rgb=True, root='./data', size=224, mmap_mode='r')
        return train_dataset, test_dataset, nb_classes
    elif args.dataset == 'Dementia':
        nb_classes = 4
        drive_dir = getattr(args, 'dataset_dir', None)
        downloader = DementiaDatasetDownloader(drive_dir=drive_dir)
        train_dir, val_dir = downloader.get_dataset()
        print(f"Dementia dataset is available at: train={train_dir} val={val_dir}")
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
        return train_dataset, test_dataset, nb_classes
    else:
        raise NotImplementedError()
    
    full_dataset = datasets.ImageFolder(root=data_dir)  # Load without transform
    # Verify the total number of images matches the sum of the splits
    assert train_size + val_size + test_size == len(full_dataset), "The sum of the splits must equal the total number of images"

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply the transformations
    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_dataset.indices)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), val_dataset.indices)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), test_dataset.indices)

    print("Number of the class = %d" % nb_classes)

    return train_dataset, test_dataset, nb_classes


def build_transform(args):
    t_train = []
    # this should always dispatch to transforms_imagenet_train
    t_train.append(transforms.RandomResizedCrop(224))
    t_train.append(transforms.AugMix(alpha= 0.4))
    #t_train.append(transforms.Lambda(lambda image: image.convert('RGB')))
    t_train.append(transforms.RandomHorizontalFlip(p=0.4))
    t_train.append(transforms.ToTensor())
    t_train.append(transforms.Normalize(mean=[.5], std=[.5]))
        

    t_test = []
    t_test.append(transforms.Resize((224, 224)))
    #t_test.append(transforms.Lambda(lambda image: image.convert('RGB')))
    t_test.append(transforms.ToTensor())
    t_test.append(transforms.Normalize(mean=[.5], std=[.5]))
    return transforms.Compose(t_train), transforms.Compose(t_test)