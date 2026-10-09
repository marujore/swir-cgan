import os

import torch
import numpy as np
import rasterio
from tqdm import tqdm
from scipy.signal.windows import tukey

from models import UNetGenerator


def get_hyperparameters(band):
    if band == 5:  # B11
        return {
            'LR_G': 0.000487, 'LR_D': 0.000011, 'DROPOUT': 0.065, 
            'L1': 13.627, 'SSIM': 42.781, 'GAN': 0.019, 
            'GRAD': 133.933, 'PERC': 1.213
        }
    elif band == 6:  # B12
        return {
            'LR_G': 0.000499, 'LR_D': 0.000010, 'DROPOUT': 0.129, 
            'L1': 19.556, 'SSIM': 34.991594, 'GAN': 0.025, 
            'GRAD': 180.192, 'PERC': 6.532
        }

def get_blend_window(patch_size, alpha=0.5):
    """
    Generates a 2D Tukey window: constant central plateau with smooth decay
    only at the overlapping edges, avoiding central brightness variations.
    """
    w1d = tukey(patch_size, alpha=alpha)
    w2d = np.outer(w1d, w1d)
    return np.clip(w2d, 1e-4, 1.0).astype(np.float32)


def load_inference_model(target_band, checkpoint_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), pad_input=16):
    """Loads the generator weights directly from the production checkpoint."""
    print(f"[*] Loading model weights from: {checkpoint_path}")
    
    hp = get_hyperparameters(target_band)
    dropout_rate = hp['DROPOUT']
    
    model = UNetGenerator(dropout_rate=dropout_rate, pad_input=pad_input).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'G_state_dict' in checkpoint:
        state_dict = checkpoint['G_state_dict']
    elif isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
    else:
        state_dict = checkpoint
        
    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[+] Generator (Band {target_band}, Dropout={dropout_rate}) loaded successfully.")
    return model


def predict_raster(model, input_raster_path, output_raster_path, device, 
                   patch_size=128, overlap=64, batch_size=32):
    print(f"[*] Processing input raster: {input_raster_path}")
    
    with rasterio.open(input_raster_path) as src:
        meta = src.meta.copy()
        image = src.read().astype(np.float32)  # Shape: (C, H, W)
        
        # 1. IDENTIFY EXACT EDGE MASK (No block cropping)
        nodata_val = src.nodata if src.nodata is not None else -9999.0
        
        # A pixel is invalid if ALL bands are nodata, or if it contains NaN
        invalid_mask = np.all(image == nodata_val, axis=0) | np.any(np.isnan(image), axis=0)
        valid_mask = ~invalid_mask
        
        # 2. CRITICAL SANITIZATION (Avoids "Infection" of Patches at the edges)
        # Replaces backgrounds and NaNs with 0.0 BEFORE the neural network processes. 
        # This ensures the U-Net does not return mathematical garbage in the boundary blocks.
        image[:, invalid_mask] = 0.0
        image = np.nan_to_num(image, nan=0.0)
        
        channels, height, width = image.shape
        if channels != 4:
            raise ValueError(f"The model expects 4 WFI bands, but it has {channels}.")
            
        max_val = float(np.max(image))
        scaled_input = False
        if max_val > 1.0:
            print(f"[*] Values exceed 1.0. Adjusting factor 10000.0 to [0, 1].")
            image = image / 10000.0
            scaled_input = True
            
        image = np.clip(image, 0.0, 1.0)
        
        # 3. OMNIDIRECTIONAL PADDING (Protects the extremities)
        # Adds reflection on ALL sides. This way the extremities of the original image
        # will always fall in the CENTER of the U-Net's field of view.
        pad_t = overlap
        pad_l = overlap
        
        height_padded = height + pad_t
        width_padded = width + pad_l
        
        stride = patch_size - overlap
        pad_b = (stride - ((height_padded - patch_size) % stride)) % stride if height_padded >= patch_size else patch_size - height_padded
        pad_r = (stride - ((width_padded - patch_size) % stride)) % stride if width_padded >= patch_size else patch_size - width_padded
        
        image = np.pad(image, ((0, 0), (pad_t, pad_b), (pad_l, pad_r)), mode='reflect')
        
        proc_h, proc_w = image.shape[1], image.shape[2]
            
        output_image = np.zeros((proc_h, proc_w), dtype=np.float32)
        weight_map = np.zeros((proc_h, proc_w), dtype=np.float32)
        blend_weights = get_blend_window(patch_size, alpha=0.5)
        
        y_starts = list(range(0, proc_h - patch_size + 1, stride))
        x_starts = list(range(0, proc_w - patch_size + 1, stride))
        coords = [(y, x) for y in y_starts for x in x_starts]
        total_patches = len(coords)
        
        print(f"[*] Total patches: {total_patches} (Stride: {stride}px | Overlap: {overlap}px)")
        
        with torch.no_grad():
            for i in tqdm(range(0, total_patches, batch_size), desc="Batch Inference"):
                batch_coords = coords[i:i + batch_size]
                
                batch_patches = [
                    image[:, y:y + patch_size, x:x + patch_size] 
                    for y, x in batch_coords
                ]
                
                batch_tensor = torch.from_numpy(np.stack(batch_patches, axis=0)).to(device)
                
                with torch.autocast(device_type=device.type):
                    preds = model(batch_tensor)
                
                preds = preds.squeeze(1).float().cpu().numpy()
                
                for (y, x), pred_patch in zip(batch_coords, preds):
                    output_image[y:y + patch_size, x:x + patch_size] += pred_patch * blend_weights
                    weight_map[y:y + patch_size, x:x + patch_size] += blend_weights

        weight_map[weight_map == 0] = 1.0
        final_output = output_image / weight_map
        
        # 4. CROP THE EXTRAPOLATED PADDING (Returns to exact original size)
        final_output = final_output[pad_t : pad_t + height, pad_l : pad_l + width]
        final_output = np.clip(final_output, 0.0, 1.0)
        
        if scaled_input:
            final_output = final_output * 10000.0
            
        # 5. FINAL TREATMENT: 3 Decimal Places and Perfect NoData Restoration
        final_output = np.round(final_output, 3)
        final_output[~valid_mask] = nodata_val  # Reapplies the nodata mask without blocks
            
        meta.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw',
            'nodata': nodata_val
        })
        
        os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(final_output.astype(np.float32), 1)
            
    print(f"[+] Final raster successfully saved at: {output_raster_path}")


def generate_swir1(input_raster, output_raster, BATCH_SIZE=32):
    DEVICE, CHECKPOINT_B11, _ = globals()
    generator_model = load_inference_model(
        target_band=5,
        checkpoint_path=CHECKPOINT_B11
    )

    predict_raster(
        model=generator_model,
        input_raster_path=INPUT_RASTER,
        output_raster_path=OUTPUT_RASTER,
        device=DEVICE,
        patch_size=128,
        overlap=64,
        batch_size=BATCH_SIZE
    )

def generate_swir2(input_raster, output_raster, BATCH_SIZE=32):
    DEVICE, _, CHECKPOINT_B12 = globals()
    generator_model = load_inference_model(
        target_band=6,
        checkpoint_path=CHECKPOINT_B12
    )

    predict_raster(
        model=generator_model,
        input_raster_path=INPUT_RASTER,
        output_raster_path=OUTPUT_RASTER,
        device=DEVICE,
        patch_size=128,
        overlap=64,
        batch_size=BATCH_SIZE
    )

def globals():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_B11 = '/tower/marujo/Downloads/Alisson/best_model_B11.pth'
    CHECKPOINT_B12 = '/tower/marujo/Downloads/Alisson/best_model_B12.pth'

    return DEVICE, CHECKPOINT_B11, CHECKPOINT_B12


if __name__ == "__main__":
    globals()

    INPUT_RASTER = '/tower/marujo/Downloads/Alisson/input/stacked.tif'
    OUTPUT_RASTER = '/tower/marujo/Downloads/Alisson/test_B12_local.tif'
    
    BATCH_SIZE = 32
    
    generate_swir1(input_raster=INPUT_RASTER, output_raster=OUTPUT_RASTER, BATCH_SIZE=BATCH_SIZE) #Informing Batch_size
    generate_swir2(input_raster=INPUT_RASTER, output_raster=OUTPUT_RASTER) #Not Informing Batch_size. 32 is default value
