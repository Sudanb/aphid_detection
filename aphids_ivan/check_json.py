import json
for split in ['train', 'val']:
    path = rf'C:\Users\sudanb\Desktop\rtdetr\aphids_ivan\dataset\annotations\instances_{split}.json'
    with open(path) as f:
        d = json.load(f)
    print(f"{split}: {len(d['images'])} images, {len(d['annotations'])} annotations")