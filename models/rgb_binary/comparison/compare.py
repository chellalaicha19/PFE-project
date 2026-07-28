"""
Solar Panel Binary Classifier - Model Comparison Script
Compares MobileNetV3-Small vs EfficientNet-B0 using your downloaded files.
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ====================== CONFIG ======================
RESULTS_DIR = Path("/Users/mac/Documents/PFE/rgb_binary_class")   # Change if your folder name is different

# File names from your ls
MOBILENET_PT = "/Users/mac/Documents/PFE/rgb_binary_class/best_model_mobileNet.pt"          # or best_model_mobileNEtEnhanced.pt
EFFICIENTNET_PT = "/Users/mac/Documents/PFE/rgb_binary_class/best_model_EfficientNet-B0.pt"

MOBILENET_JSON = "/Users/mac/Documents/PFE/rgb_binary_class/final_results_mobileNet.json"
EFFICIENTNET_JSON = "/Users/mac/Documents/PFE/rgb_binary_class/final_results_EfficientNet-B0.json"

MOBILENET_CONF = "/Users/mac/Documents/PFE/rgb_binary_class/confusion_matrix.png"
EFFICIENTNET_CONF = "/Users/mac/Documents/PFE/rgb_binary_class/confusion_matrix_EfficientNet-B0.png"

print("🔍 Loading comparison data...\n")

# Load JSON results
with open(RESULTS_DIR / MOBILENET_JSON) as f:
    mob_results = json.load(f)

with open(RESULTS_DIR / EFFICIENTNET_JSON) as f:
    eff_results = json.load(f)

# Model sizes (from your ls -la)
mob_size_mb = (RESULTS_DIR / MOBILENET_PT).stat().st_size / (1024 * 1024)
eff_size_mb = (RESULTS_DIR / EFFICIENTNET_PT).stat().st_size / (1024 * 1024)

print("="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"{'Metric':<20} {'MobileNetV3-Small':>18} {'EfficientNet-B0':>18} {'Winner':>12}")
print("-"*80)
print(f"{'Test Accuracy':<20} {mob_results['accuracy']:18.4f} {eff_results['accuracy']:18.4f} "
      f"{'MobileNet' if mob_results['accuracy'] > eff_results['accuracy'] else 'EfficientNet':>12}")
print(f"{'Macro F1':<20} {mob_results['f1']:18.4f} {eff_results['f1']:18.4f} "
      f"{'MobileNet' if mob_results['f1'] > eff_results['f1'] else 'EfficientNet':>12}")
print(f"{'ROC-AUC':<20} {mob_results.get('auc', 0):18.4f} {eff_results.get('auc', 0):18.4f} "
      f"{'MobileNet' if mob_results.get('auc', 0) > eff_results.get('auc', 0) else 'EfficientNet':>12}")
print(f"{'Model Size (MB)':<20} {mob_size_mb:18.2f} {eff_size_mb:18.2f} "
      f"{'MobileNet':>12} (much smaller)")
print("="*80)

winner = "MobileNetV3-Small" if mob_results['f1'] > eff_results['f1'] else "EfficientNet-B0"
print(f"🏆 **Overall Winner by Macro F1**: {winner}\n")

# ====================== VISUAL COMPARISON ======================
# Create figure with 3 subplots (2 rows, 2 columns - but we'll use gridspec for better control)
fig = plt.figure(figsize=(15, 10))

# Performance metrics comparison (top-left)
metrics = ['Accuracy', 'Macro F1', 'ROC-AUC']
mob_values = [mob_results['accuracy'], mob_results['f1'], mob_results.get('auc', 0)]
eff_values = [eff_results['accuracy'], eff_results['f1'], eff_results.get('auc', 0)]

x = np.arange(len(metrics))
width = 0.35

ax1 = plt.subplot(2, 2, 1)
bars1 = ax1.bar(x - width/2, mob_values, width, label='MobileNetV3-Small', color='#1f77b4')
bars2 = ax1.bar(x + width/2, eff_values, width, label='EfficientNet-B0', color='#ff7f0e')

ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Value labels
for i, (v1, v2) in enumerate(zip(mob_values, eff_values)):
    ax1.text(i - width/2, v1 + 0.01, f'{v1:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax1.text(i + width/2, v2 + 0.01, f'{v2:.4f}', ha='center', fontsize=9, fontweight='bold')

# ====================== NEW: Model Size Comparison Bar Chart ======================
ax2 = plt.subplot(2, 2, 2)

# Model size comparison
model_names = ['MobileNetV3-Small', 'EfficientNet-B0']
model_sizes = [mob_size_mb, eff_size_mb]
colors = ['#1f77b4', '#ff7f0e']

bars = ax2.bar(model_names, model_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for i, (bar, size) in enumerate(zip(bars, model_sizes)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{size:.2f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add size reduction percentage
    if i == 0:  # MobileNet
        reduction = ((eff_size_mb - mob_size_mb) / eff_size_mb) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'▼ {reduction:.1f}% smaller', ha='center', va='center', 
                 fontsize=9, color='white', fontweight='bold')

ax2.set_ylabel('Model Size (MB)')
ax2.set_title('Model Size Comparison')
ax2.grid(True, alpha=0.3, axis='y')

# Add horizontal line for reference
ax2.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10 MB reference')
ax2.legend()

# Set y-axis limit with some padding
ax2.set_ylim(0, max(model_sizes) * 1.15)

# ====================== Confusion Matrices Side by Side ======================
# MobileNet confusion
ax3 = plt.subplot(2, 2, 3)
img_mob = plt.imread(RESULTS_DIR / MOBILENET_CONF)
ax3.imshow(img_mob)
ax3.axis('off')
ax3.set_title('MobileNetV3-Small\nConfusion Matrix', fontsize=11)

# EfficientNet confusion
ax4 = plt.subplot(2, 2, 4)
img_eff = plt.imread(RESULTS_DIR / EFFICIENTNET_CONF)
ax4.imshow(img_eff)
ax4.axis('off')
ax4.set_title('EfficientNet-B0\nConfusion Matrix', fontsize=11)

plt.suptitle('Model Comparison: MobileNetV3-Small vs EfficientNet-B0', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison_completeEnhanced.png", dpi=200, bbox_inches='tight')
plt.show()

print(f"📊 Complete comparison chart saved as: {RESULTS_DIR}/model_comparison_completeEnhanced.png")

# ====================== Additional Standalone Size Comparison ======================
# Create a separate detailed size comparison figure
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart with percentage labels
bars = ax5.bar(model_names, model_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_ylabel('Model Size (MB)', fontsize=12)
ax5.set_title('Model Size Comparison (Standalone)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for bar, size in zip(bars, model_sizes):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{size:.2f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add efficiency ratio text
size_ratio = eff_size_mb / mob_size_mb
ax5.text(0.5, -0.15, f'EfficientNet is {size_ratio:.1f}x larger than MobileNet', 
         ha='center', transform=ax5.transAxes, fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Pie chart for size distribution
ax6.pie(model_sizes, labels=model_names, colors=colors, autopct='%1.1f%%', 
        startangle=90, explode=(0.05, 0), shadow=True)
ax6.set_title('Size Distribution', fontsize=14, fontweight='bold')

plt.suptitle('Storage Efficiency Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_size_comparison_detailedEnhanced.png", dpi=200)
plt.show()

print(f"📊 Detailed size comparison saved as: {RESULTS_DIR}/model_size_comparison_detailedEnhanced.png")

# ====================== Summary Statistics ======================
print("\n" + "="*80)
print("STORAGE EFFICIENCY ANALYSIS")
print("="*80)
print(f"MobileNetV3-Small size:    {mob_size_mb:.2f} MB")
print(f"EfficientNet-B0 size:      {eff_size_mb:.2f} MB")
print(f"Size difference:           {abs(eff_size_mb - mob_size_mb):.2f} MB")
print(f"MobileNet is {eff_size_mb/mob_size_mb:.1f}x smaller than EfficientNet")
print(f"Storage savings:           {(1 - mob_size_mb/eff_size_mb)*100:.1f}%")
print("="*80)

# Performance vs Size analysis
print("\n" + "="*80)
print("PERFORMANCE VS SIZE ANALYSIS")
print("="*80)

# Efficiency score: Accuracy per MB
mob_eff = mob_results['accuracy'] / mob_size_mb
eff_eff = eff_results['accuracy'] / eff_size_mb

print(f"MobileNet:   {mob_results['accuracy']*100:.2f}% accuracy @ {mob_size_mb:.2f} MB")
print(f"             → {mob_eff:.4f} accuracy per MB")
print(f"EfficientNet:{eff_results['accuracy']*100:.2f}% accuracy @ {eff_size_mb:.2f} MB")
print(f"             → {eff_eff:.4f} accuracy per MB")
print(f"MobileNet is {mob_eff/eff_eff:.1f}x more efficient (accuracy per MB)")

if mob_results['accuracy'] > eff_results['accuracy']:
    print("\n✅ MobileNet wins on both accuracy AND size!")
elif abs(mob_results['accuracy'] - eff_results['accuracy']) < 0.02:
    print("\n📊 Models have similar accuracy, but MobileNet is much smaller → Choose MobileNet!")
else:
    print("\n⚠️ Trade-off: Better performance vs smaller size")