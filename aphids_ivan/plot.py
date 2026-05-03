# plot_metrics.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

log_file = Path(r"C:\Users\sudanb\Desktop\rtdetr\RT-DETR\rtdetrv2_pytorch\output\rtdetrv2_r50vd_m_6x_coco\log.txt")

epochs, losses, ap50, ap5095, ar100 = [], [], [], [], []

with open(log_file) as f:
    for line in f:
        try:
            d = json.loads(line.strip())
            if 'epoch' not in d: continue
            epochs.append(d['epoch'])
            losses.append(d['train_loss'])
            bbox = d.get('test_coco_eval_bbox', [])
            ap5095.append(bbox[0] if len(bbox) > 0 else 0)
            ap50.append(bbox[1] if len(bbox) > 1 else 0)
            ar100.append(bbox[8] if len(bbox) > 8 else 0)
        except:
            continue

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('RT-DETR Aphid Detection Training', fontsize=14)

axes[0].plot(epochs, losses, 'b-', linewidth=1.5)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, ap5095, 'g-', label='AP@0.50:0.95', linewidth=1.5)
axes[1].plot(epochs, ap50, 'r-', label='AP@0.50', linewidth=1.5)
axes[1].set_title('Average Precision')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AP')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, ar100, 'm-', linewidth=1.5)
axes[2].set_title('Average Recall @100')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AR')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to training_metrics.png")