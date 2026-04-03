import os, shutil, random

name_map = {
    "mild_dementia": "MildDemented",
    "moderated_dementia": "ModerateDemented",
    "non_demented": "NonDemented",
    "very_mild_demented": "VeryMildDemented"
}

base = "AlzheimersCV/archive"
train_dir = os.path.join(base, "train_images")
val_dir = os.path.join(base, "val_images")
VAL_SPLIT = 0.15

#Rename train folders to match test
for old, new in name_map.items():
    old_path = os.path.join(train_dir, old)
    new_path = os.path.join(train_dir, new)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

#Carve out val split from train
for cls in os.listdir(train_dir):
    cls_train = os.path.join(train_dir, cls)
    cls_val = os.path.join(val_dir, cls)
    os.makedirs(cls_val, exist_ok=True)

    images = os.listdir(cls_train)
    random.shuffle(images)
    n_val = int(len(images) * VAL_SPLIT)

    for img in images[:n_val]:
        shutil.move(os.path.join(cls_train, img), os.path.join(cls_val, img))

